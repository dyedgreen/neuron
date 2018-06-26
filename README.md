# Neuron
Neuron is a simple neural network, built from scratch in c++. It is not
optimized in any way, so I would recommend against using it for anything
important.

## How to use neuron:
You can initialize a network object that has a specified amount of hidden layers and
a specified layer size:
```c++
#include <stdlib>
#include "network.h"

int main() {
  // Init a network with N hidden layers of size HIDDEN, an input size of IN and an
  // output size of OUT (<N, IN, OUT, HIDDEN>). Initialize to random values between +/-5.
  neuron::Network<float, 2, 2, 1, 4> net = new neuron::Network<float, 2, 2, 1, 4>(5);

  float data[2] = {1, 2};
  float *result = net->run(data);

  std::cout << "Output: " << *result << std::endl;

  return 0;
}
```

**TODO:** Better example here + fix backprop!