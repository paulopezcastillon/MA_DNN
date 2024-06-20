
import tensorflow as tf
import tensorflow_datasets as tfds
import tf_keras as keras
import tensorflow_model_optimization as tfmot
import numpy as np

import sys
import math

arguments = sys.argv

data, metadata = tfds.load(arguments[2], as_supervised=True, with_info=True)

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
model = tf.keras.models.load_model('model/'+arguments[1]+'.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations=[tf.lite.Optimize.DEFAULT]

def representative_dataset():
    for images, labels in train_data.take(100):
        yield [tf.dtypes.cast(images, tf.float32)]
      
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

# Save the TFLite model
with open('model/'+arguments[1]+'_int8.tflite', 'wb') as f:
    f.write(tflite_model)
