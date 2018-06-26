#ifndef H_NETWORK_
#define H_NETWORK_

#include <cstdlib>

namespace neuron {

// Network template
template<typename T, int n, int in, int out, int hidden>
class Network {
private:
  int sizes[n];
  T mat_in[hidden][in];
  T mat_hidden[n][hidden][hidden];
  T mat_out[out][hidden];
  T bias[n+1][hidden];
  T state[n+1][hidden];
  T result[out];
public:
  bool useActivation;
  bool useBias;
  Network(T);
  ~Network();
  void load(T*, T*, T*, T*);
  T* run(T[in]);
  void backprop(T[out], T);
  void print();
  void dump();
};

} // End neuron namespace

#endif