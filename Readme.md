## Multi-layer perceptron for MNIST dataset. 

### Motivation


More than anything, I built this project to take a closer look at how neural networks approximates a function. 

During my Postgraduate degree I was given an introduction to various machine learning algorithms both supervised and unsupervised, I found neural networks to be the most fascinating. We read and replicated a few papers, however, most of the lab work was done using libraries such as Tensorflow and PyTorch. 

Although those experiments were helpful in learning high level concepts like overfitting or what is a learning rate, I found them to be insufficient in giving me a lower level of understanding. For instance, I did understand that forward propagation (for weights) is a dot product between the output of the last layer (or input) and weights of the following layer. However, I could not understand the underlying matrices, their dimensions and so on. 

Another reason for implementing this repository is 3Blue1Brown. When I first watched Grants lectures on neural networks, as a somewhat established software engineer, I thought, this is so simple I can just code this. However, as I sat down to code those explanations, I realised that making sense out of those videos is one thing but coding them is another. 

I am sure there must be many people who would have tried to replicate 3B1B’s work in a form of a library, this is just my take on it. 

If any of aformentioned resonates with you, this repository is for you! 

Whilst this project does not implement all the bells and whistles one might find in the latest and greatest libraries i.e., PyTorch or alikes, it removes many abstractions and in a way allows beginners to inspect the computations manually. 

Lastly, the thing that helped me most was actually creating a small arbitrary neural network and multiplying matrices manually. I have included this small neural network in the form of a unit test in this repository called `unit.py`. 


## Limitations

There are multiple ways of for doing backpropogation, two of them are
1. SGD
2. Mini-batch

### SGD
In this version, weights are updated for each sample.

### Mini-Batch
Mini-batch updates weights once for multiple samples.

Although this neural network tries to replicate 3b1b's lecture on neural network, this does not implement mini-batch, it instead implements SGD. 

## Network Instance

`network.py` is the core class which allows a user to train and test a model. An instance of a network class can be used to add layers and activation functions within the layers as follows. 

Network instance requires three parameters

1. `id` - `string` value which is used to identify the network uniquely. This allows storing and retrieving the relevant parameters for the network i.e., weights and biases. 

2. `layers` - `list<Layer>` this is used to store the information about layers in the network and their parameters

3. `random` -  `boolean` flag to toggle between random parameter initialisation and stored parameters. If `False` the network will use the stored parameters. 

#### Example

`network = Network( 'mnist', [Layer(neurons = 16,activation=relu)], True )`

## Dependencies

There is only one dependency for this project and every other dependency is handled automatically. 

This package requires:

`Docker` - `v4.53.0 or above`


## Usage

a. Clone the repository

b. Change your current working directory to the repository 

c. In a shell terminal run `./docker-init.sh`

Assuming that the docker daemon is running, this script will create a container and will start the shell of that container. 

`main.py` contains examples of how to train and test the model. 

`network.py` contains train and test methods which can be extendend for experimentations. 

## Tests

Within the container, use the following command to run the unit tests. 

```python3 unit.py``` 

The package includes the following tests. 
- `test_forward_pass` - to validate if the forward pass is doing the dot products correctly
- `test_backpropogation` - to validate if the partial derivatives for the cross entropy are getting computed correctly. 

These tests assumes the biases are `0`.

