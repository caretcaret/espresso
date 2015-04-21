#ifndef ESPRESSO_COMPONENTS_H
#define ESPRESSO_COMPONENTS_H

#include <iostream>
#include <vector>
#include "Halide.h"

namespace Espresso {

/**************/
/* Base class */
/**************/

class Layer {
public:
  Halide::Func forward;
  /* Dimensions of the component output. Operations on
   * fewer than 4 dimensions perform by batching up
   * dimensions w, z, and/or y.
   * TODO: make sure this works efficiently with memory layout */
  int x, y, z, w;

  Layer(int x, int y, int z, int w)
    : forward("forward"), x(x), y(y), z(z), w(w), i("i"), j("j"), k("k"), l("l") {}

protected:
  /* Variables that are used everywhere */
  Halide::Var i, j, k, l;
};

/*****************/
/* Vision layers */
/*****************/

class Convolution : public Layer {
public:
  Convolution(Layer input, Halide::Func kernel, int kernel_x, int kernel_y, int pad_x, int pad_y, int stride_x, int stride_y)
    : Layer(input.x - kernel_x + 1, input.y - kernel_y + 1, input.z, input.w) {
    // TODO: fix
    Halide::RDom r(0, kernel_x, 0, kernel_y);
    Halide::RVar m = r.x;
    Halide::RVar n = r.y;

    forward(i, j, k, l) = Halide::sum(Halide::sum(input.forward(i - m, j - n, k, l) * kernel(m, n)));
  }
};

class MaxPool : public Layer {
public:
  MaxPool(Layer input, int pool_x, int pool_y, int pad_x, int pad_y, int stride_x, int stride_y)
    : Layer(input.x, input.y, input.z, input.w) {
    // assume pool_x and pool_y are odd
    Halide::RDom r(-pool_x / 2, pool_x / 2, -pool_y / 2, pool_y / 2);
    Halide::RVar m = r.x;
    Halide::RVar n = r.y;

    forward(i, j, k, l) = Halide::maximum(input.forward(i + m, j + n, k, l));
  }
};

// TODO: combine into single pooling class
class AvgPool : public Layer {
public:
  AvgPool(Layer input, int pool_x, int pool_y)
    : Layer(input.x, input.y, input.z, input.w) {
    // assume pool_x and pool_y are odd
    Halide::RDom r(-pool_x / 2, pool_x / 2, -pool_y / 2, pool_y / 2);
    Halide::RVar m = r.x;
    Halide::RVar n = r.y;

    forward(i, j, k, l) = Halide::sum(input.forward(i + m, j + n, k, l)) / (pool_x * pool_y);
  }
};

// TODO: LRN
// TODO: im2col

/***************/
/* Loss layers */
/***************/
// TODO: Softmax loss
// TODO: Euclidean loss
// TODO: Hinge loss
// TODO: Sigmoid cross entropy loss
// TODO: Infogain loss
// TODO: Accuracy/top-k

/*********************/
/* Activation layers */
/*********************/

class ReLU : public Layer {
public:
  ReLU(Layer input, float negative_slope=0.f)
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

/***************/
/* Data layers */
/***************/

class MemoryData : public Layer {
public:
  /* By convention x = width, y = height, z = channels, w = batch size */
  MemoryData(Halide::ImageParam input, int input_x, int input_y, int input_z, int input_w)
    : Layer(input_x, input_y, input_z, input_w) {
    forward(i, j, k, l) = input(i, j, k, l);
  }
};

// TODO

/*****************/
/* Common layers */
/*****************/

class InnerProduct : public Layer {
public:
  /* W: output_x * (input.x + 1), where the last column is the bias
   * or W: output_x * input.x for no bias term */
  /* TODO: use blas gemv/gemm */
  InnerProduct(Layer input, Halide::Func W, int output_x, bool bias_term=true)
    : Layer(output_x, input.y, input.z, input.w) {
    Halide::RDom r(0, input.x);
    Halide::Func b;

    b(i) = W(i, input.x);
    if (bias_term) {
      forward(i, j, k, l) = Halide::sum(W(i, r.x) * input.forward(r.x, j, k, l)) + b(i);
    } else {
      forward(i, j, k, l) = Halide::sum(W(i, r.x) * input.forward(r.x, j, k, l));
    }
  }
};

class Softmax : public Layer {
public:
  Softmax(Layer input, float epsilon=1e-8)
    : Layer(input.x, input.y, input.z, input.w) {
    Halide::Func activation("activation");
    Halide::Func normalizer("normalizer");
    Halide::RDom r(0, input.x);

    activation(i, j, k, l) = Halide::fast_exp(input.forward(i, j, k, l));
    normalizer(j, k, l) = Halide::sum(activation(r.x, j, k, l)) + epsilon;
    forward(i, j, k, l) = activation(i, j, k, l) / normalizer(j, k, l);
  }
};

// TODO: splitting
// TODO: flattening
// TODO: concatenation
// TODO: slicing
// TODO: eltwise
// TODO: argmax
// TODO: MVN

} // namespace Espresso

#endif // ESPRESSO_COMPONENTS_H
