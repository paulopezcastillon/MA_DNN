
import tensorflow as tf
import tensorflow_datasets as tfds
import tf_keras as keras
import tensorflow_model_optimization as tfmot
import numpy as np
import sys
import math

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


arguments = sys.argv

model = tf.keras.models.load_model('model/'+arguments[1]+'.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations=[tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_types=[tf.float16]

tflite_model = converter.convert()

# Save the TFLite model
with open('model/'+arguments[1]+'_float16.tflite', 'wb') as f:
    f.write(tflite_model)

# input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# interpreter.set_tensor(input_details[0]['index'], input_data)

# interpreter.invoke()

# output_data = interpreter.get_tensor(output_details[0]['index'])

# print()
# print(output_data)

#PRUNING

#print("pruning the model!")

# pruning_params = {
#       'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
#                                                                final_sparsity=0.50,
#                                                                begin_step=0,
#                                                                end_step=1000)
# }

# model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

# model_for_pruning.compile(
#     optimizer='adam',
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=['accuracy']
# )

# prune_callbacks = [
#     tfmot.sparsity.keras.UpdatePruningStep(),
#     tfmot.sparsity.keras.PruningSummaries(log_dir='log')
#   ]

# history_pruned = model_for_pruning.fit(train_data, epochs=10, steps_per_epoch= math.ceil(n_train/BATCH_SIZE), callbacks=prune_callbacks)