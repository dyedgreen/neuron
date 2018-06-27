# Neuron
Neuron is a simple neural network, built from scratch in C++. It is not
optimized in any way and 'from scratch' means it doesn't use the `math.h`
C library, so I would recommend against using it for anything serious.

## Hello World Example
You can initialize a network object that has a specified amount of hidden layers and
a specified layer size:
```C++
#include <iostream>
#include "network.h"

int main() {
  srand(12345);

  // Init a network with N hidden layers of size HIDDEN, an input size of IN and an
  // output size of OUT (<N, IN, OUT, HIDDEN>). Initialize to random values between +/-5.
  neuron::Network<float, 1, 2, 1, 3> *net = new neuron::Network<float, 1, 2, 1, 3>(5);

  // We will train this to evaluate an X-OR
  float train[4][2] = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
  };
  float target[4][1] = {{0},{1},{1},{0}};
  float *res;

  // Heuristic gradient descend
  for (int i = 0, n; i < 5e4; i ++) {
    res = net->run(train[i % 4]);
    net->backprop(target[i % 4], 0.4);
  }

  // Print a human readable network to the console
  net->print();
  std::cout << std::endl;

  // Show us what you got...
  for (int i = 0; i < 4; i ++) {
    res = net->run(train[i]);
    std::cout << "Case {" <<train[i][0]<<", "<<train[i][1]<< "}: " << *res << " (should be " << target[i][0] << ")" << std::endl;
  }

  return 0;
}
```
The above example can be found at `helloworld.cpp` and compiled with `g++ helloworld.cpp network.cpp -o hello`
(assuming you want to use the GNU compiler).

## The `Network` Class
To use neuron, include the `"network.h"` header in your C++ file. The `Network` class is a template
accessible from the namespace `neuron` and offers the following api:

#### Template Parameters
The template parameters determine the size of the network.
```C++
neuron::Network<typename numeric_type, int number_of_hidden_layers, int input_size, int output_size, int hidden_size>

Example:
neuron::Network<float, 2, 2, 1, 4>
```
The above example creates a network with two hidden layers of size 4 that takes two inputs and has one output.

#### Runtime Configuration
You can tell a network to exclude bias or the use of it's activation function.
```C++
network->use_activation = false; // Default is true
network->use_bias = false; // Default is true
```

#### Run and Backprop
The network can be executed on an array input. The input is expected to be of the correct size. As an
output, the network returns a pointer that point to consecutive numbers of the networks numeric type.
```C++
float input[3] = {1, 2, 3};
float *result;

result = network->run(input);

for (i = 0; i < output_size; i ++) {
  std::cout << result[i] << std::endl;
}
```
After the network has run, you can call `backprop()` to execute a gradient descend step. **Calling
`backprop()` before running the network at least once is undefined behavior**.
```C++
float target[3] = {1, 2, 3};
float learning_rate = 0.4;

network->backprop(target, learning_rate);
```
**When you call run several times, the values held by the previously returned pointers will be
overwritten.**

#### Print and Dump
You can print the networks learned vales to the console. Print prints a human-readable output, while
dump generates an output that is formatted as C++ array literals for easy copy-pasting.
```C++
network->print();
network->dump();
```

## Training
Neuron is very simple, so all it can do is heuristic gradient descend, using a batch size of one.
On the plus side, training has a friendly memory footprint. To train a network, call the `backprop()`
function on your network object after calling `run()` on the appropriate training example:
```C++
float learning_rate = 0.1;

float input[]  = {1, 2, ...};
float target[] = {1, 2, ...};

net->run(input);
net->backprop(target, learning_rate);
```

## Architecture
Neuron implements a simple NN with a variable number hidden layers, which are fully connected and
all have the same size. It uses a sigmoid as it's activation function. All hidden layers have a
bias and the activation function is applied for all layers with the exclusion of the output layer.