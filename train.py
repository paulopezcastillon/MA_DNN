

import tensorflow as tf
import tensorflow_datasets as tfds
import tf_keras as keras
import tensorflow_model_optimization as tfmot
import numpy as np

import math

data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

train_data, test_data = data['train'], data['test']

class_names = metadata.features['label'].names

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

#model
model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28,1)),
  tf.keras.layers.Dense(50, activation=tf.nn.relu),
  tf.keras.layers.Dense(50, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#compile
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

BATCH_SIZE = 32

train_data = train_data.repeat().shuffle(n_train).batch(BATCH_SIZE)

history = model.fit(train_data, epochs=1000, steps_per_epoch= math.ceil(n_train/BATCH_SIZE))

model.save("model/model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations=[tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types=[tf.float32]

tflite_model = converter.convert()

# Guardar el modelo TFLite en un archivo
tflite_model_path = 'model/model_float32.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)