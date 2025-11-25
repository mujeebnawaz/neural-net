from layer import *
import os.path
import sys

class Network:
	def __init__(self, id, layers, random=False): 

		self.layers = layers
		self.id = id
		self.random = random

		## Load the weights
		if not random:
			if os.path.isfile("data/weights/"+self.id+".npz"):
				self.weights = np.load("data/weights/"+self.id+".npz")
			if os.path.isfile("data/bias/"+self.id+".npz"):
				self.bias = np.load("data/bias/"+self.id+".npz")

	def train(self, training_data, training_labels, lr=0.01):
		for i, x in enumerate(training_data):
			output = self.forward_pass(x)
			true = get_one_hot(training_labels[i], 10)
			delta = output - true   
			self.backprop(delta=delta) 
			cross = cross_entropy_loss( true, output )
			print(cross)
			if cross < 1 and i > 40000:
				if not os.path.isfile("data/weights/"+self.id+".npz"):
					arrays_to_save = [layer.weights for layer in self.layers]  
					np.savez("data/weights/"+self.id+".npz", *arrays_to_save) 

				if not os.path.isfile("data/bias/"+self.id+".npz"):
					arrays_to_save = [layer.bias for layer in self.layers]  
					np.savez("data/bias/"+self.id+".npz", *arrays_to_save) 
		
	
	def get_weights(self, index, neurons=0, input=0):
		if not self.random:
			return self.weights["arr_" + str(index)]
		else:
			n_in = len(input)
			n_out = neurons
			limit = np.sqrt(6 / (n_in + n_out))
			return np.random.uniform(-limit, limit, size=(n_out, n_in)).astype(np.float32)

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
	
	def backprop(self, delta, lr=0.001, out=[]):
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

