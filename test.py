import tensorflow as tf
import tensorflow_datasets as tfds
import tf_keras as keras
import tensorflow_model_optimization as tfmot
import numpy as np
import time

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

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=arguments[1])
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

total_time=0.0
for test_image, test_label in test_data:
    test_image = tf.expand_dims(test_image, axis=0)  # Ensure the input shape matches (1, 28, 28, 1)
    interpreter.set_tensor(input_details[0]['index'], test_image)
    start = time.time()
    interpreter.invoke()
    total_time+=(time.time()-start)
    output = interpreter.get_tensor(output_details[0]['index'])
    accuracy.update_state(test_label, output)

print("TFLite model accuracy:", accuracy.result().numpy(), "time: ", total_time)
