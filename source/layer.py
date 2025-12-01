from functional import *
import numpy as np
from data.data import *


class Layer:
	
	def __init__(self, neurons=1, activation=mock, type=None):
		self.neurons = neurons
		self.activation = activation
		self.type = type
    
	def __str__(self):
		return f"-- Neurons: {self.neurons}, Activation: {self.activation.__name__}, Save: {self.save} --"

	def forward(self, inputs):
		if not hasattr(self, 'weights') or not hasattr(self, 'bias'):
			return
		self.inputs = inputs
		output = np.dot( self.weights, self.inputs ) + self.bias
		self.pre_activation = output 
		activation = self.activation( output )
		self.output = activation
		return activation

	"""
	Properties 
	"""
	@property 
	def activation(self):
		return self._activation
	
	@activation.setter
	def activation(self, func):
		if not callable(func):
			raise ValueError("Name of activation function should be a callable function")
		
		self._activation = func

	@property
	def weights(self):
		return self._weights

	@weights.setter
	def weights(self, generated_weights):
		if len( generated_weights.shape ) != 2 :
			raise ValueError("Weights must be 2-dimensional (neurons x inputs)")
		
		self._weights = generated_weights
		
	@property
	def bias(self):
		return self._bias

	@bias.setter
	def bias(self, generated_bias):
		if len(  generated_bias.shape ) != 1 :
			raise ValueError("Biases must be 1-dimensional (number of neurons)")

		self._bias = generated_bias

	@property
	def neurons(self):
		return self._neurons
	
	@neurons.setter
	def neurons(self, no_of_neurons):
		if not no_of_neurons:
			raise ValueError('Number of Neurons cannot be empty')

		self._neurons = no_of_neurons

	@property
	def inputs(self):
		return self._inputs
	
	@inputs.setter
	def inputs(self, input_list):
		if not isinstance(input_list, np.ndarray):
			raise ValueError("Inputs must be provided as a list")

		if len(input_list) == 0 or len(input_list) >= 1000:
			raise ValueError("Number of inputs must be greater than 0 and less than 1000")

		self._inputs = input_list


