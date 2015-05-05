#ifndef ESPRESSO_LAYERS_ACTIVATION_H
#define ESPRESSO_LAYERS_ACTIVATION_H

#include "Halide.h"
#include "layer.h"

namespace Espresso {

/*********************/
/* Activation layers */
/*********************/

class ReLU : public Layer {
public:
  ReLU(Layer input, float negative_slope=0.0f)
    : Layer(input.x, input.y, input.z, input.w) {
    forward(i, j, k, l) = Halide::max(input.forward(i, j, k, l), 0)
      + negative_slope * Halide::min(input.forward(i, j, k, l), 0);
  }
};

class Sigmoid : public Layer {
public:
  Sigmoid(Layer input)
    : Layer(input.x, input.y, input.z, input.w) {
    forward(i, j, k, l) = 1 / (1 + Halide::fast_exp(-input.forward(i, j, k, l)));
  }
};

class Tanh : public Layer {
public:
  Tanh(Layer input)
    : Layer(input.x, input.y, input.z, input.w) {
    forward(i, j, k, l) = Halide::tanh(input.forward(i, j, k, l));
  }
};

class Abs : public Layer {
public:
  Abs(Layer input)
    : Layer(input.x, input.y, input.z, input.w) {
    forward(i, j, k, l) = Halide::abs(input.forward(i, j, k, l));
  }
};

class Power : public Layer {
public:
  Power(Layer input, float power=1.0f, float scale=1.0f, float shift=0.0f)
    : Layer(input.x, input.y, input.z, input.w) {
    forward(i, j, k, l) = Halide::fast_pow(input.forward(i, j, k, l) * scale + shift, power);
  }
};

class BNLL : public Layer {
public:
  BNLL(Layer input)
    : Layer(input.x, input.y, input.z, input.w) {
    forward(i, j, k, l) = Halide::fast_log(1 + Halide::fast_exp(input.forward(i, j, k, l)));
  }
};

class Threshold : public Layer {
public:
  Threshold(Layer input, float threshold)
    : Layer(input.x, input.y, input.z, input.w) {
    forward(i, j, k, l) = Halide::select(input.forward(i, j, k, l) <= threshold, 0, 1);
  }
};

class Dropout : public Layer {
public:
  Dropout(Layer input, bool train=false, float p=0.5)
    : Layer(input.x, input.y, input.z, input.w) {
    if (train) {
      forward(i, j, k, l) = Halide::select(Halide::random_float() > p, input.forward(i, j, k, l) / (1 - p), 0);
    } else {
      forward(i, j, k, l) = input.forward(i, j, k, l);
    }
  }
};

} // namespace Espresso

#endif // ESPRESSO_LAYERS_ACTIVATION_H
