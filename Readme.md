## Multi-layer perceptron for MNIST dataset.

More than anything, I built this project for taking a closer look at how neural networks approximates a function. 

Prior to this I had used Tensorflow and PyTorch during my PostGrad and although I gained a good level of understanding of deep learning - I quite did not understand backpropogation and feed forward on a very basic level.

Whilst this project does not implement all the bells and whistles one might find in the latest and greatest libraries i.e., PyTorch or alikes, it removes many abstractions and sort of allows beginners to inspect the computations manually. 

## Dependencies

There is only one dependency for this project and every other dependency is handled automatically. 

This package requires:

`Docker` - `v4.53.0 or above`

## Tests

Use the following commmand to run the unit tests. 

```python3 unit.py``` 

The package includes the following tests. 
- test_forward_pass - to validate if the forward pass is doing the dot products correctly
- test_backpropogation - to validate if the partial derivatives for the cross entropy are getting computed correctly. 

These tests assumes the biases are 0.

Both of the above mentioned tests are included in `unit.py`


## Usage

`main.py` contains the example of how this package can be used. 

### Network Instance
An instance of a network class can be used to add layers and activation functions within the layers as follows. 

Network instance requires three parameters

1. `ID_OF_THE_NETWORK` - A `string` value which is used to identify the network uniquely. This allows storing and retrieving the relevant parameters for the network i.e., weights and biases. 

2. `Layers` - A list of type `Layer`