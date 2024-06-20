import tensorflow as tf
import tensorflow_datasets as tfds
import tf_keras as keras
import tensorflow_model_optimization as tfmot
import numpy as np
import time

import sys
import math
import os

import zipfile
import tempfile

arguments = sys.argv

def calculate_model_size(model):
    total_params = model.count_params()
    size_in_bytes = total_params * 4  # float32, which is 4 bytes
    return size_in_bytes


model = tf.keras.models.load_model('model/'+arguments[1]+'.h5')

_, keras_file = tempfile.mkstemp('.h5')
keras.models.save_model(model, keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)

data, metadata = tfds.load(arguments[2], as_supervised=True, with_info=True)

train_data, test_data = data['train'], data['test']

n_train = metadata.splits["train"].num_examples
n_test = metadata.splits["test"].num_examples

#Normalize input data
def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

train_data = train_data.map(normalize)
test_data = test_data.map(normalize)

train_data = train_data.cache()
test_data = test_data.cache()

BATCH_SIZE=32

train_data = train_data.repeat().shuffle(n_train).batch(BATCH_SIZE)

start = time.time()
test_loss, test_accuracy = model.evaluate(test_data.batch(32), verbose=False)
end = time.time()
print(f"Keras model accuracy: {test_accuracy}, time: {end - start}")

end_step = np.ceil(60000 / BATCH_SIZE).astype(np.int32) * 10

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.5,
                                                               final_sparsity=0.95,
                                                               begin_step=0,
                                                               end_step=1000)
}

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

model_for_pruning.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model_for_pruning.summary()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir="log"),
]

history_pruned = model_for_pruning.fit(train_data, epochs=25, steps_per_epoch= math.ceil(n_train/BATCH_SIZE), callbacks=callbacks)

start = time.time()
test_loss, test_accuracy = model_for_pruning.evaluate(test_data.batch(32), verbose=False)
end = time.time()

print(f"Keras model accuracy: {test_accuracy}, time: {end - start}")

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

_, pruned_keras_file = tempfile.mkstemp('.h5')
keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
pruned_tflite_model = converter.convert()

_, pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(pruned_tflite_file, 'wb') as f:
  f.write(pruned_tflite_model)

print('Saved pruned TFLite model to:', pruned_tflite_file)

with open("model/"+arguments[1]+"_pruned.tflite", 'wb') as f:
  f.write(pruned_tflite_model)

#remove 0 weigths
def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

print("Size of gzipped baseline Keras model: %.2f bytes" % ((get_gzipped_model_size(keras_file)/1024)))
print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
print("Size of gzipped pruned TFlite model: %.2f bytes" % ((get_gzipped_model_size(pruned_tflite_file)/1024)))