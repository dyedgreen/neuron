#include <iostream>
#include "network.h"

int main() {
  // 2 -> 4 -> 4 -> 1 neural network
  neuron::Network<float, 1, 2, 1, 3> *net = new neuron::Network<float, 1, 2, 1, 3>(10);
  //net->useActivation = false;
  net->useBias = false;
  net->print();
  // net->dump();

  float train[4][2] = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
  };
  float target[4][1] = {{0},{1},{1},{0}}; // X-OR
  float *res;

  std::cout << "\n\n";

  for (int i = 0, n; i < 1000; i ++) {
    n = i % 4;
    res = net->run(train[n]);
    //if (i % 5000 == 0) std::cout << "Iteration " << i << ": " << *res << std::endl;
    net->backprop(target[n], 0.1);
  }

  std::cout << "\n\n";

  for (int i = 0; i < 4; i ++) {
    res = net->run(train[i]);
    std::cout << "Case " << i << ": " << *res << " (should be " << target[i][0] << ")" << std::endl;
  }

  std::cout << "\n\n";

  net->print();

  std::cout << "\n";

  // std::cout << neuron::exp<float>(1) << std::endl << std::endl;

  // Load a network
  // neuron::Network<float, 2, 2, 1, 4> *net2 = new neuron::Network<float, 2, 2, 1, 4>(false);
  // float mat_in[] = {7.82637e-06, 0.131538, 0.755605, 0.45865, 0.532767, 0.218959, 0.0470446, 0.678865};
  // float mat_hidden[] = {0.679296, 0.934693, 0.383502, 0.519416, 0.830965, 0.0345721, 0.0534616, 0.5297, 0.671149, 0.00769819, 0.383416, 0.0668422, 0.417486, 0.686773, 0.588977, 0.930436, 0.846167, 0.526929, 0.0919649, 0.653919, 0.415999, 0.701191, 0.910321, 0.762198, 0.262453, 0.0474645, 0.736082, 0.328234, 0.632639, 0.75641, 0.991037, 0.365339};
  // float mat_out[] =  {0.247039, 0.98255, 0.72266, 0.753356};
  // float bias[] = {0.303037, -0.854628, 0.263269, 0.769414, -0.45458, -0.127177, 0.53299, -0.0445365, -0.524451, -0.450186, -0.28147, -0.666986};
  // net2->load((float*)&mat_in, (float*)&mat_hidden, (float*)&mat_out, (float*)&bias);

  // net2->print();
  // res = net->run(in);
  // std::cout << *res << std::endl;

  // Clean up
  delete net;
  // delete net2;
}