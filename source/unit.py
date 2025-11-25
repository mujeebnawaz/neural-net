from layer import *
from data.datamnist import *
from functional import *
from network import Network

"""
Imagine a neural network with 

1 input 1 hidden and one output layer

do not apply any activation functions on these layers

input layer will have two neurons

hidden layer will have 3 neurons with following weights
[2,6],[4,2],[1,9]

and output layer will have 2 neurons with following weights
[1,3,3],[4,2,6]

If input is 
[2,3]

what would be the output of feed forward?

"""

class Unit:

    def __init__(self): 
        # x = np.array([-1,1])
        # y = np.array([22,14,29])
        # print(np.outer(x,y))
        self.init_weights()
        self.init_bias()
        self.network = Network(
                'unit',
                [ Layer(neurons = 3, activation=relu), Layer(neurons = 2, type="output", activation=softmax) ],
                False
        )

    def init_weights(self):
        if not os.path.isfile("data/weights/unit.npz"):
            np.savez("data/weights/unit.npz", np.array([[2,6],[4,2],[1,9]], dtype=float), np.array([[1,3,3],[4,2,6]], dtype=float))
    
    def init_bias(self):
        if not os.path.isfile("data/bias/unit.npz"):
            np.savez("data/bias/unit.npz", np.zeros(3), np.zeros(2))

unit = Unit()
unit.network.train(np.array([[2,3]]),[0])