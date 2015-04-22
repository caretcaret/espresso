#ifndef ESPRESSO_COMPONENTS_H
#define ESPRESSO_COMPONENTS_H

#include <iostream>
#include <vector>
#include <stdexcept>
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
   * dimensions w, z, and/or y. */
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
  Convolution(Layer input, Halide::Func kernel, int kernel_x, int kernel_y, int n_filters, int pad_x=0, int pad_y=0, int stride_x=1, int stride_y=1)
    : Layer(input.x + (2 * pad_x - kernel_x) / stride_x + 1, input.y + (2 * pad_y - kernel_y) / stride_y + 1, n_filters, input.w) {
    // Kernel size must be kernel_x by kernel_y by input.z by n_filters, where n_filters is the number of filters
    // TODO: fix
    Halide::RDom r(0, kernel_x, 0, kernel_y);
    Halide::RVar m = r.x;
    Halide::RVar n = r.y;

    forward(i, j, k, l) = Halide::sum(input.forward(i - m, j - n, k, l) * kernel(m, n));
  }
};

class Pooling : public Layer {
public:
  Pooling(Layer input, int pool_x, int pool_y, int pad_x=0, int pad_y=0, int stride_x=1, int stride_y=1, std::string method="max")
    : Layer(input.x + (2 * pad_x - pool_x) / stride_x + 1, input.y + (2 * pad_y - pool_y) / stride_y + 1, input.z, input.w) {
    Halide::Func rand_x, rand_y;
    Halide::RDom r(-pool_x / 2, pool_x / 2 + 1, -pool_y / 2, pool_y / 2 + 1);
    Halide::RVar m = r.x;
    Halide::RVar n = r.y;
    // TODO: fix

    if (method == "max") {
      forward(i, j, k, l) = Halide::maximum(input.forward(i + m, j + n, k, l));
    } else if (method == "average") {
      forward(i, j, k, l) = Halide::sum(input.forward(i + m, j + n, k, l)) / (pool_x * pool_y);
    } else if (method == "stochastic") {
      rand_x(i, j, k, l) = Halide::random_int() % pool_x - pool_x / 2;
      rand_y(i, j, k, l) = Halide::random_int() % pool_y - pool_y / 2;
      forward(i, j, k, l) = input.forward(i + rand_x(i, j, k, l), j + rand_y(i, j, k, l), k, l);
    } else {
      throw new std::invalid_argument("No such pooling method");
    }
  }
};

class LRN : public Layer {
public:
  LRN(Layer input, int region_x=1, int region_y=1, int region_z=1, float alpha=1.0f, float beta=5.0f)
    : Layer(input.x, input.y, input.z, input.w) {
    // across_channels => region_x and region_y = 1; within_channels => region_z = 1
    Halide::Func activation("activation");
    Halide::Func normalizer("normalizer");
    Halide::Func padded("padded");
    Halide::RDom r(-region_x / 2, region_x / 2 + 1, -region_y / 2, region_y / 2 + 1, -region_z / 2, region_z / 2 + 1);

    padded(i, j, k, l) = 0;
    padded(i, j, k, l) = input.forward(i, j, k, l);

    activation(i, j, k, l) = Halide::sum(padded(i + r.x, j + r.y, k + r.z, l));
    normalizer(i, j, k, l) = Halide::fast_pow(1 + (alpha / (region_x * region_y * region_z)) * activation(i, j, k, l), beta);
    forward(i, j, k, l) = activation(i, j, k, l) / normalizer(i, j, k, l);
  }
};

/***************/
/* Loss layers */
/***************/

class MultinomialLogisticLoss : public Layer {
public:
  MultinomialLogisticLoss(Layer pred, Layer labels)
    : Layer(1, 1, 1, 1) {
    // predictions are probability vectors num_classes by 1 by 1 by batch_size
    // labels are integers 1 by 1 by 1 by batch_size
    Halide::RDom r(0, pred.w);

    forward(i, j, k, l) = -Halide::sum(Halide::fast_log(pred.forward(labels.forward(0, 0, 0, r.x), 0, 0, r.x))) / pred.w;
  }
};

class EuclideanLoss : public Layer {
public:
  EuclideanLoss(Layer pred, Layer obs)
    : Layer(1, 1, 1, 1) {
    // predictions and observations are 1 by 1 by 1 by batch_size
    Halide::Func diff;
    Halide::RDom r(0, pred.w);

    diff(w) = pred.forward(0, 0, 0, w) - obs.forward(0, 0, 0, w);
    forward(i, j, k, l) = Halide::sum(diff(r.x) * diff(r.x)) / (2 * pred.w);
  }
};

class InfoGainLoss : public Layer {
public:
  InfoGainLoss(Layer pred, Layer labels, Halide::Func infogain)
    : Layer(1, 1, 1, 1) {
    // infogain is a n_class by n_class matrix
    Halide::RDom r(0, pred.x, 0, pred.w);
    Halide::RVar cls = r.x;
    Halide::RVar w = r.y;

    forward(i, j, k, l) = -Halide::sum(infogain(labels.forward(0, 0, 0, w), cls) * Halide::fast_log(pred.forward(cls, 0, 0, w))) / pred.w;
  }
};

class Accuracy : public Layer {
public:
  Accuracy(Layer pred, Layer labels, int top_k=1)
    : Layer(1, 1, 1, 1) {
    Halide::Func correct;
    Halide::RDom r(0, pred.x);
    Halide::RDom s(0, pred.w);

    if (top_k != 1) {
      // TODO
      throw new std::invalid_argument("Top-k not implemented for k != 1");
    }

    correct(l) = Halide::argmax(r, pred.forward(r.x, 0, 0, l))[0] == labels.forward(0, 0, 0, l);
    forward(i, j, k, l) = Halide::sum(Halide::select(correct(s.x), 1, 0)) / pred.w;
  }
};

class HingeLoss : public Layer {
public:
  HingeLoss(Layer pred, Layer labels, int p)
    : Layer(1, 1, 1, 1) {
    Halide::Func loss;
    Halide::RDom r(0, pred.x, 0, pred.w);

    loss(i, l) = Halide::max(0, 1 + Halide::select(labels.forward(0, 0, 0, l) == i, -1, 1) * pred.forward(i, 0, 0, l));
    forward(i, j, k, l) = Halide::sum(Halide::fast_pow(loss(r.x, r.y), p)) / pred.w;
  }
};

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
  Softmax(Layer input)
    : Layer(input.x, input.y, input.z, input.w) {
    Halide::Func activation("activation");
    Halide::Func normalizer("normalizer");
    Halide::RDom r(0, input.x);

    activation(i, j, k, l) = Halide::fast_exp(input.forward(i, j, k, l));
    normalizer(j, k, l) = Halide::sum(activation(r.x, j, k, l));
    forward(i, j, k, l) = activation(i, j, k, l) / normalizer(j, k, l);
  }
};

class Argmax : public Layer {
public:
  Argmax(Layer input, int top_k=1)
    : Layer(1, 1, 1, input.w) {
    // input is a probability vector x by 1 by 1 by w
    Halide::RDom r(0, input.x);

    if (top_k != 1) {
      // TODO
      throw new std::invalid_argument("top-k for argmax not implemented");
    }

    forward(i, j, k, l) = Halide::argmax(r, input.forward(r.x, j, k, l))[0];
  }
};

class Split : public Layer {
public:
  Split(Layer input)
    : Layer(input.x, input.y, input.z, input.w) {
    // TODO: currently splitting is modeled as a no-op until backprop is needed
    forward(i, j, k, l) = input.forward(i, j, k, l);
  }
};

class Flatten : public Layer {
public:
  Flatten(Layer input)
    : Layer(input.x * input.y * input.z, 1, 1, input.w) {
      forward(i, j, k, l) = input.forward((i / (input.y * input.z)), (i / input.z) % input.y, i % input.z, l);
  }
};

// TODO: concatenation
// TODO: slicing
// TODO: eltwise
// TODO: MVN

} // namespace Espresso

#endif // ESPRESSO_COMPONENTS_H
