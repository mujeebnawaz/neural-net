from layer import *
from data.datamnist import *
from functional import *
from network import Network
		
network = Network('mnist',
			[
				Layer(neurons = 16, activation=relu),
				Layer(neurons = 16, activation=relu),
				Layer(neurons = 10, type="output", activation=softmax)
			]
			,True
		)

data_mnist = Data_MNIST()
(training_data, training_labels), (testing_data, testing_labels) = data_mnist.get_mnist()

network.train(testing_data, testing_labels)
