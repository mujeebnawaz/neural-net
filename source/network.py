from layer import *
import os.path
import matplotlib.pyplot as plt

class Network:
	def __init__(self, id, layers, random=False): 

		self.layers = layers
		self.id = id
		self.random = random

		# If parameters are not 
		if not random:
			self.load_stored_params()

	def train(self, training_data, training_labels, error_threshold = 1, min_sample=40000, number_of_classes = 10):
		"""
		This training loop implements backpropogation for cross entropy.
		For any other cost function or regression, please extend this class and implment 
		train() and backprop()

		Args:
            training_data (np.ndarray): Input data from the dataset i.e., set of pixels from MNIST dataset.
            training_labels (np.ndarray): Set of true labels for the dataset.

        Returns:
            None
		"""
		for sample_index, input in enumerate(training_data):
			output = self.forward_pass(input)
			true = get_one_hot(training_labels[sample_index], number_of_classes)
			delta = output - true
			self.backprop(delta=delta) 
			cross = cross_entropy_loss( true, output )
			if cross < error_threshold and sample_index > min_sample:
				if not os.path.isfile("data/weights/"+self.id+".npz"):
					arrays_to_save = [layer.weights for layer in self.layers]  
					np.savez("data/weights/"+self.id+".npz", *arrays_to_save) 

				if not os.path.isfile("data/bias/"+self.id+".npz"):
					arrays_to_save = [layer.bias for layer in self.layers]  
					np.savez("data/bias/"+self.id+".npz", *arrays_to_save) 

	def test(self, testing_data, testing_labels, number_of_classes = 10):
		# Automatically turn off the random weights initialisation if testing. 
		self.random = False
		# Load the sotred parameters
		self.load_stored_params()
		for sample_index, input in enumerate(testing_data):
			output = self.forward_pass(input)
			true = get_one_hot(testing_labels[sample_index], number_of_classes)

			cross = cross_entropy_loss( true, output )
			print(cross)

		...

	def load_stored_params(self):
		weights_path = f"data/weights/{self.id}.npz"
		bias_path = f"data/bias/{self.id}.npz"

		if os.path.isfile(weights_path) and os.path.isfile(bias_path):
			self.weights = np.load(weights_path, allow_pickle=True)
			self.bias = np.load(bias_path, allow_pickle=True)
		else:
			raise FileNotFoundError(
				f"Model parameters do not exist for ID '{self.id}'. "
				"Please train the model first."
			)
			

	def get_weights(self, index, neurons=0, input=0):
		if not self.random:
			return self.weights["arr_" + str(index)]
		else:
			return get_random([len(input), neurons])

	def get_bias(self, index, neurons=0, input=0):
		if not self.random:
			return self.bias["arr_"+str(index)]
		else:
			return np.zeros(neurons) 

	def forward_pass(self, input):
		output = input.flatten()
		for index, layer in enumerate(self.layers):
			if not hasattr(layer, 'weights') or not hasattr(layer, 'bias'):
				layer.weights = self.get_weights(index, layer.neurons, output)
				layer.bias = self.get_bias(index, layer.neurons, output)
			output = layer.forward(output)

		return output

	def backprop(self, delta, lr=0.001):
		reversed_layers = list(reversed(self.layers))
		for index, layer in enumerate(reversed_layers):
			if layer.type == 'output':
				layer.weights -= lr * np.outer( delta, reversed_layers[index+1].output ) 
				layer.bias -= lr * delta
			
			else:
				delta = ( reversed_layers[index-1].weights.T @ delta ) 
				relu_prime = relu_derivative( layer.pre_activation )
				delta_hidden = relu_prime * delta
				layer.weights -= lr * np.outer( delta_hidden, layer.inputs)
				layer.bias -= lr * delta_hidden

