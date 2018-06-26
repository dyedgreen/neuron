#include <iostream>
#include "network_.h"

namespace neuron {


template<typename T>
T fac(T n) {
  T res = (T)1;
  for (; n > 0; n --) {
    res *= n;
  }
  return res;
}

template<typename T>
T pow(T x, int n) {
  T res = (T)1;
  for (; n > 0; n --) {
    res *= x;
  }
  return res;
}

template<typename T>
T exp(T x) {
  T res = (T)1;
  for (int i = 1; i < 32; i ++) {
    res += pow<T>(x, i) / fac<T>(i);
  }
  return res;
}

template<typename T>
T sig(T x) {
  return (T)1 / ((T)1 + exp<T>(-x));
}

template<typename T>
T randRange(T max, bool posNeg) {
  if (posNeg) {
    return (T)( 2 * (double)max * (double)rand() / (double)RAND_MAX - (double)max );
  }
  return (T)( (double)max * (double)rand() / (double)RAND_MAX );
}


template<typename T, int n, int in, int out, int hidden>
Network<T, n, in, out, hidden>::Network(T max) {
  for (int i = 0, j; i < hidden; i ++) {
    for (j = 0; j < in; j ++) {
      this->mat_in[i][j] = max ? randRange<T>(max, true) : (T)0;
    }
  }
  for (int l = 0, i, j; l < n; l ++) {
    for (i = 0; i < hidden; i ++) {
      for (j = 0; j < hidden; j ++) {
        this->mat_hidden[l][i][j] = max ? randRange<T>(max, true) : (T)0;
      }
    }
  }
  for (int i = 0, j; i < out; i ++) {
    for (j = 0; j < hidden; j ++) {
      this->mat_out[i][j] = max ? randRange<T>(max, true) : (T)0;
    }
  }
  for (int i = 0, j; i < n+1; i ++) {
    for (j = 0; j < hidden; j ++) {
      this->bias[i][j] = max ? randRange<T>(max, true) : (T)0;
    }
  }
  this->useActivation = true;
  this->useBias = true;
}

template<typename T, int n, int in, int out, int hidden>
Network<T, n, in, out, hidden>::~Network() {
  // Nothing to do here...
}

template<typename T, int n, int in, int out, int hidden>
void Network<T, n, in, out, hidden>::load(T *mat_in, T *mat_hidden, T *mat_out, T *bias) {
  // The pointers are expected to be arrays of T, that follow the layout of Network::dump()
  // Input matrix
  for (int i = 0; i < hidden*in; i ++) {
    this->mat_in[i/in][i%in] = mat_in[i];
  }
  // Hidden matrices
  for (int i = 0; i < hidden*hidden*n; i ++) {
    this->mat_hidden[i/hidden/hidden][(i/hidden)%hidden][i%hidden] = mat_hidden[i];
  }
  // Output matrix
  for (int i = 0; i < out*hidden; i ++) {
    this->mat_out[i/hidden][i%hidden] = mat_out[i];
  }
  // Bias
  for (int i = 0; i < hidden*(n+1); i ++) {
    this->bias[i/hidden][i%hidden] = bias[i];
  }
}

template<typename T, int n, int in, int out, int hidden>
T* Network<T, n, in, out, hidden>::run(T input[in]) {
  // In
  for (int i = 0, j; i < hidden; i ++) {
    // Matrix
    this->state[0][i] = (T)0;
    for (j = 0; j < in; j ++) {
      this->state[0][i] += input[j] * this->mat_in[i][j];
    }
    // Bias
    if (this->useBias) {
      this->state[0][i] += this->bias[0][i];
    }
    // Activation
    if (this->useActivation) {
      this->state[0][i] = sig<T>(this->state[0][i]);
    }
  }
  // Hidden
  for (int l = 0, i, j; l < n; l ++) {
    for (i = 0; i < hidden; i ++) {
      // Matrix
      this->state[l+1][i] = (T)0;
      for (j = 0; j < hidden; j ++) {
        this->state[l+1][i] += this->state[l][j] * this->mat_hidden[l][i][j];
      }
      // Bias
      if (this->useBias) {
        this->state[l+1][i] += this->bias[l+1][i];
      }
      // Activation
      if (this->useActivation) {
        this->state[l+1][i] = sig<T>(this->state[l+1][i]);
      }
    }
  }
  // Out
  for (int i = 0, j; i < out; i ++) {
    // Matrix
    this->result[i] = (T)0;
    for (j = 0; j < hidden; j ++) {
      result[i] += this->state[n][j] * this->mat_out[i][j];
    }
    // No bias / activation on output
  }
  return &(this->result[0]);
}

template<typename T, int n, int in, int out, int hidden>
void Network<T, n, in, out, hidden>::backprop(T target[out], T learningRate) {
  // Find gradient w.r.t. wights + do gradient descend.
  // ====
  // In practice: Go backwards using the transpose of the matrices, the fact
  // that sig' = (1 - sig) * sig, and the chain rule. For the weights this means:
  // FeedTo * FeedFrom = weight, where FeedTo is propagated backwards.
  // TODO: Add in bias (right now, this does not work with bias...)
  // Out
  for (int i = 0, j; i < hidden; i ++) {
    T tmp = (T)0;
    for (j = 0; j < out; j ++) {
      // Backprop error
      tmp += this->mat_out[j][i] * (this->result[j] - target[j]);
      // Do gradient descend (hence -)
      this->mat_out[j][i] -= learningRate * (this->result[j] - target[j]);
    }
    this->state[n][i] = tmp * this->state[n][i] * (1 - this->state[n][i]);
  }
  // Hidden
  for (int l = n-1, i, j; l >= 0; l --) {
    for (int i = 0, j; i < hidden; i ++) {
      T tmp = (T)0;
      for (j = 0; j < hidden; j ++) {
        // Backprop error
        tmp += this->mat_hidden[l][j][i] * this->state[l+1][i];
        // Do gradient descend (hence -)
        this->mat_hidden[l][j][i] -= learningRate * this->state[l+1][i];
      }
      this->state[l][i] = tmp * this->state[l][i] * (1 - this->state[l][i]);
    }
  }
  // Input
  for (int i = 0, j; i < in; i ++) {
    for (j = 0; j < hidden; j ++) {
      // No Backprop, since there are no more weights before this...
      // Do gradient descend (hence -)
      this->mat_in[j][i] -= learningRate * this->state[0][i];
    }
  }
}

template<typename T, int n, int in, int out, int hidden>
void Network<T, n, in, out, hidden>::print() {
  std::cout << "Neural Network("<<n<<", "<<in<<", "<<out<<", "<<hidden<<")\n";
  std::cout << "Input Matrix:" << std::endl;
  for (int i = 0, j; i < hidden; i ++) {
    std::cout << "  ";
    for (j = 0; j < in; j ++) {
      std::cout << "  " << this->mat_in[i][j];
    }
    std::cout << std::endl;
  }
  for (int l = 0, i, j; l < n; l ++) {
    std::cout << "Bias "<<(l+1)<<":" << std::endl << "  ";
    for (i = 0; i < hidden; i ++) {
      std::cout << "  " << this->bias[l][i];
    }
    std::cout << std::endl;
    std::cout << "Hidden Matrix "<<(l+1)<<":" << std::endl;
    for (i = 0; i < hidden; i ++) {
      std::cout << "  ";
      for (j = 0; j < hidden; j ++) {
        std::cout << "  " << this->mat_hidden[l][i][j];
      }
      std::cout << std::endl;
    }
  }
  std::cout << "Bias "<<(n+1)<<":" << std::endl << "  ";
  for (int i = 0; i < hidden; i ++) {
    std::cout << "  " << this->bias[n][i];
  }
  std::cout << std::endl;
  std::cout << "Output Matrix:" << std::endl;
  for (int i = 0, j; i < out; i ++) {
    std::cout << "  ";
    for (j = 0; j < hidden; j ++) {
      std::cout << "  " << this->mat_out[i][j];
    }
    std::cout << std::endl;
  }
}

template<typename T, int n, int in, int out, int hidden>
void Network<T, n, in, out, hidden>::dump() {
  std::cout << "[DUMP] Neural Network("<<n<<", "<<in<<", "<<out<<", "<<hidden<<")\n";
  std::cout << "Input Matrix:" << std::endl;
  std::cout << "    {";
  for (int i = 0; i < hidden*in; i ++) {
    std::cout << this->mat_in[i/in][i%in];
    if (i + 1 < hidden*in) {
      std::cout << ", ";
    }
  }
  std::cout << "}" << std::endl;
  std::cout << "Bias:" << std::endl << "    {";
  for (int i = 0; i < hidden*(n+1); i ++) {
    std::cout << this->bias[i/hidden][i%hidden];
    if (i + 1 < hidden*(n+1)) {
      std::cout << ", ";
    }
  }
  std::cout << "}" << std::endl;
  std::cout << "Hidden Matrices:" << std::endl << "    {";
  for (int i = 0; i < hidden*hidden*n; i ++) {
    std::cout << this->mat_hidden[i/hidden/hidden][(i/hidden)%hidden][i%hidden];
    if (i + 1 < hidden*hidden*n) {
      std::cout << ", ";
    }
  }
  std::cout << "}" << std::endl;
  std::cout << "Output Matrix:" << std::endl << "    {";
  for (int i = 0; i < out*hidden; i ++) {
    std::cout << this->mat_out[i/hidden][i%hidden];
    if (i + 1 < out*hidden) {
      std::cout << ", ";
    }
  }
  std::cout << "}" << std::endl;
}


} // End namespace neuron