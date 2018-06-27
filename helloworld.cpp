#include <iostream>
#include "network.h"

// Hello world example from README.

int main() {
  srand(12345);

  // Init a network with N hidden layers of size HIDDEN, an input size of IN and an
  // output size of OUT (<N, IN, OUT, HIDDEN>). Initialize to random values between +/-10.
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