from .data import Data
import numpy as np	
import os.path

class Data_MNIST(Data):
	def __init__(self):
		super().__init__('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz')
		
		# Get the dataset if it doesn't already exists!
		if not os.path.isfile(self.file):
			print('Downloading MNIST Dataset...')
			self.download()
			print(f'Dataset downloaded at {self.file}')

	def get_mnist(self):
		if not os.path.isfile(self.file):
			print("Dataset does not exist")

		with np.load(self.file) as data:
			x_train, y_train = data['x_train'], data['y_train']
			x_test, y_test = data['x_test'], data['y_test']

			# Normalise the images. 
			x_train = x_train.astype(np.float32) / 255.0
			x_test  = x_test.astype(np.float32) / 255.0

			return (x_train, y_train), (x_test, y_test)
