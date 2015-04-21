#ifndef ESPRESSO_NET_COMPONENTS_H
#define ESPRESSO_NET_COMPONENTS_H

#include <iostream>
#include <vector>
#include "Halide.h"

namespace Espresso {
/*
Rambling about how training may work:
The data is an input image of shape (# instances, feature dim1, feature dim2) and a target image of shape (# instances, output dim1, output dim2).
The input of training is an initial tuple of parameters.
The final output of training is a tuple of parameters that have been fit to the data.
To model the change in parameters over time, each parameter has a time dimension, and each instance is defined in terms of the previous. For each parameter p, p(t) is defined in terms of p(t-1), where the provided initial parameters is p(0) and the output parameters is p(t_f), where t_f is the number of epochs to run the training. This requires a static number of epochs to be specified at training init time, but if one wants a dynamic way to train, one can run some number of epochs at a time, and after each hyperepoch, check the error to see if another hyperepoch needs to be run. This is to prevent parameters from being copied between device and host repeatedly.
Specifying all parameters this way may seem inefficient, but should be made efficient with scheduling on a sliding window.
The network is a pipeline that is used as a component in training. It accepts a batch of input data of shape (# instances, feature dim1, feature dim2).
The network is built from a composition of layers. Each layer defines the transform it makes based on a vector of parameters and how to backpropagate results.
The transform takes in (# instances, feature dim1, feature dim2) with parameters and outputs (# instances, output dim1, output dim2). The programmer is responsible for making sure the dimensions match up; maybe we'll have some error checking mechanism when the layers are hooked up.
The layer is also responsible for specifying backprop derivative stuff:
for gradient descent, we need to update each parameter with param - alpha * d loss / d param. Chain rule: d/dx f(g(x)) = f'(g(x)) g'(x). Each layer fulfills the role of both f and g, so since it needs to implement df/dg * dg/dp, where df/dg = next layer's derivative, dg/dp = derivative of the current layer with respect to a parameter, each layer needs an implementation of df/dx and df/dp.
In doing so, each layer maintains a buffer for its parameters.
We can encode the loss function as a layer without params, and have the differentiation just work.

With these things, it *should* be possible to train a whole network with one kernel call (provided the amount of memory available to the gpu is sufficient).
*/

class Component {
public:
  Halide::Func forward;
  /* Dimensions of the component output. Operations on
   * fewer than 4 dimensions perform by batching up
   * dimensions w, z, and/or y.
   * TODO: make sure this works efficiently with memory layout */
  int x, y, z, w;

  Component(int x, int y, int z, int w)
    : forward("forward"), x(x), y(y), z(z), w(w) {}
};

class InputLayer : public Component {
public:
  /* By convention x = width, y = height, z = channels, w = batch size */
  InputLayer(Halide::ImageParam input, int input_x, int input_y, int input_z, int input_w)
    : Component(input_x, input_y, input_z, input_w) {
    Halide::Var i, j, k, l;
    forward(i, j, k, l) = input(i, j, k, l);
  }
};

class ReLU : public Component {
public:
  ReLU(Component input)
    : Component(input.x, input.y, input.z, input.w) {
    Halide::Var i, j, k, l;
    forward(i, j, k, l) = Halide::max(input.forward(i, j, k, l), 0);
  }
};

class InnerProduct : public Component {
public:
  /* W: output_x * (input.x + 1), where the last column is the bias */
  /* TODO: use blas gemv/gemm */
  InnerProduct(Component input, Halide::Func W, int output_x)
    : Component(output_x, input.y, input.z, input.w) {
    Halide::Var i, j, k, l;
    Halide::RDom r(0, input.x);
    Halide::Func b;

    b(i) = W(i, input.x);
    forward(i, j, k, l) = Halide::sum(W(i, r.x) * input.forward(r.x, j, k, l)) + b(i);
  }
};

class Softmax : public Component {
public:
  Softmax(Component input, float epsilon=1e-8)
    : Component(input.x, input.y, input.z, input.w) {
    Halide::Func activation("activation");
    Halide::Func normalizer("normalizer");
    Halide::Var i, j, k, l;
    Halide::RDom r(0, input.x);

    activation(i, j, k, l) = Halide::exp(input.forward(i, j, k, l));
    normalizer(j, k, l) = Halide::sum(activation(r.x, j, k, l)) + epsilon;
    forward(i, j, k, l) = activation(i, j, k, l) / normalizer(j, k, l);
  }
};

class Convolution : public Component {
public:
  Convolution(Component input, Halide::Func kernel, int kernel_x, int kernel_y, int padding_x, int padding_y, int stride_x, int stride_y)
    : Component(input.x - kernel_x + 1, input.y - kernel_y + 1, input.z, input.w) {
    // TODO: fix
    Halide::Var i, j, k, l;
    Halide::RDom r(0, kernel_x, 0, kernel_y);
    Halide::RVar m = r.x;
    Halide::RVar n = r.y;

    forward(i, j, k, l) = Halide::sum(Halide::sum(input.forward(i - m, j - n, k, l) * kernel(m, n)));
  }
};

class MaxPool : public Component {
public:
  MaxPool(Component input, int pool_x, int pool_y)
    : Component(input.x, input.y, input.z, input.w) {
    // assume pool_x and pool_y are odd
    Halide::Var i, j, k, l;
    Halide::RDom r(-pool_x / 2, pool_x / 2, -pool_y / 2, pool_y / 2);
    Halide::RVar m = r.x;
    Halide::RVar n = r.y;

    forward(i, j, k, l) = Halide::maximum(input.forward(i + m, j + n, k, l));
  }
};

class AvgPool : public Component {
public:
  AvgPool(Component input, int pool_x, int pool_y)
    : Component(input.x, input.y, input.z, input.w) {
    // assume pool_x and pool_y are odd
    Halide::Var i, j, k, l;
    Halide::RDom r(-pool_x / 2, pool_x / 2, -pool_y / 2, pool_y / 2);
    Halide::RVar m = r.x;
    Halide::RVar n = r.y;

    forward(i, j, k, l) = Halide::sum(input.forward(i + m, j + n, k, l)) / (pool_x * pool_y);
  }
};

} // namespace Espresso

#endif // ESPRESSO_NET_COMPONENTS_H
