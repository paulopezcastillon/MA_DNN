

train:
	python3 train.py

train_MNIST:
	python3 MNIST.py

quantization_all: quantization_16f quantization_MNIST_16f quantization_8int

quantization_16f:
	python3 quantization/quantization_16f.py model

quantization_MNIST_16f:
	python3 quantization/quantization_16f.py MNIST

quantization_8int:
	python3 quantization/quantization_8int.py model fashion_mnist

quantization_MNIST_8int:
	python3 quantization/quantization_8int.py MNIST mnist

test_all: test_model test_16f test_8int

test_model:
	python3 test.py model/model_float32.tflite fashion_mnist

test_MNIST:
	python3 test.py model/MNIST_float32.tflite mnist

test_16f:
	python3 test.py model/model_float16.tflite fashion_mnist

test_MNIST_16f:
	python3 test.py model/MNIST_float16.tflite mnist

test_8int:
	python3 test.py model/model_int8.tflite fashion_mnist

test_MNIST_8int:
	python3 test.py model/MNIST_int8.tflite mnist

pruning_model:
	python3 pruning/pruning.py model fashion_mnist

pruning_MNIST:
	python3 pruning/pruning.py MNIST mnist

test_pruning:
	python3 test.py model/model_pruned.tflite fashion_mnist

test_MNIST_pruning:
	python3 test.py model/MNIST_pruned.tflite mnist