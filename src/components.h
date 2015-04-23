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
  /* Dimensions of the layer output. Operations on
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
  Convolution(Layer input, Halide::Func kernel, int kernel_x, int kernel_y, int n_filters,
      int pad_x=0, int pad_y=0, int stride_x=1, int stride_y=1, bool bias_term=true, int group=1)
    : Layer((input.x + 2 * pad_x - kernel_x) / stride_x + 1,
            (input.y + 2 * pad_y - kernel_y) / stride_y + 1,
            n_filters,
            input.w) {
    // Kernel size is kernel_x by kernel_y by input.z by n_filters / group, where n_filters is the number of filters,
    // and +1 on input.z if bias is used; bias is stored at 0, 0, input.z for each filter.
    // kernel_x, kernel_y must be odd
    Halide::Func padded("padded");
    Halide::Func convolved("convolved");
    Halide::Func bias("bias");
    Halide::RDom r(-kernel_x / 2, kernel_x / 2 + 1, -kernel_y / 2, kernel_y / 2 + 1, 0, input.z);

    padded(i, j, k, l) = 0;
    padded(i * 2 * pad_x, j * 2 * pad_y, k, l) = input.forward(i, j, k, l);

    convolved(i, j, k, l) = Halide::sum(input.forward(i + r.x, j + r.y, r.z, l) * kernel(r.x + kernel_x / 2, r.y + kernel_y / 2, r.z, k / group));
    if (bias_term) {
      bias(k) = kernel(0, 0, input.z, k / group);
      convolved(i, j, k, l) += bias(k);
    }

    forward(i, j, k, l) = convolved(i * stride_x, j * stride_y, k, l);
  }
};

class Pooling : public Layer {
public:
  Pooling(Layer input, std::string method, int pool_x, int pool_y,
      int pad_x=0, int pad_y=0, int stride_x=1, int stride_y=1)
    : Layer((input.x + 2 * pad_x - pool_x) / stride_x + 1,
            (input.y + 2 * pad_y - pool_y) / stride_y + 1,
            input.z,
            input.w) {
    Halide::Func padded("padded");
    Halide::Func pooled("pooled");
    Halide::Func rand_x, rand_y;
    Halide::RDom r(-pool_x / 2, pool_x / 2 + 1, -pool_y / 2, pool_y / 2 + 1);

    padded(i, j, k, l) = 0;
    padded(i * 2 * pad_x, j * 2 * pad_y, k, l) = input.forward(i, j, k, l);

    if (method == "max") {
      pooled(i, j, k, l) = Halide::maximum(padded(i + r.x, j + r.y, k, l));
    } else if (method == "average") {
      pooled(i, j, k, l) = Halide::sum(padded(i + r.x, j + r.y, k, l)) / (pool_x * pool_y);
    } else if (method == "stochastic") {
      rand_x(i, j, k, l) = Halide::random_int() % pool_x - pool_x / 2;
      rand_y(i, j, k, l) = Halide::random_int() % pool_y - pool_y / 2;
      pooled(i, j, k, l) = padded(i + rand_x(i, j, k, l), j + rand_y(i, j, k, l), k, l);
    } else {
      throw new std::invalid_argument("No such pooling method");
    }

    forward(i, j, k, l) = pooled(i * stride_x, j * stride_y, k, l);
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
    Halide::Func bias("bias");

    forward(i, j, k, l) = Halide::sum(W(i, r.x) * input.forward(r.x, j, k, l));
    if (bias_term) {
      bias(i) = W(i, input.x);
      forward(i, j, k, l) += bias(i);
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

class Eltwise : public Layer {
public:
  Eltwise(Layer input1, Layer input2, std::string op="sum")
    : Layer(input1.x, input1.y, input1.z, input1.w) {
    if (op == "sum") {
      forward(i, j, k, l) = input1.forward(i, j, k, l) + input2.forward(i, j, k, l);
    } else if (op == "prod") {
      forward(i, j, k, l) = input1.forward(i, j, k, l) * input2.forward(i, j, k, l);
    } else if (op == "max") {
      forward(i, j, k, l) = Halide::max(input1.forward(i, j, k, l), input2.forward(i, j, k, l));
    } else {
      throw new std::invalid_argument("Operation not supported in Eltwise");
    }
  }
};

class Concat : public Layer {
public:
  Concat(std::initializer_list<Layer> inputs, int axis=0)
    // not the actual dimensions; the axis of concatenation is set again below
    : Layer(inputs.begin()->x, inputs.begin()->y, inputs.begin()->z, inputs.begin()->w) {
    // layers may only be concatenated along one axis, all other axes must have the same dimensions

    int offset = 0;

    // TODO: how to make this cleaner >.<
    if (axis == 0) {
      for (Layer input : inputs) {
        Halide::RDom r(0, input.x);
        forward(offset + r, j, k, l) = input.forward(r, j, k, l);
        offset += input.x;
      }
      x = offset;
    } else if (axis == 1) {
      for (Layer input : inputs) {
        Halide::RDom r(0, input.y);
        forward(i, offset + r, k, l) = input.forward(i, r, k, l);
        offset += input.y;
      }
      y = offset;
    } else if (axis == 2) {
      for (Layer input : inputs) {
        Halide::RDom r(0, input.z);
        forward(i, j, offset + r, l) = input.forward(i, j, r, l);
        offset += input.z;
      }
      z = offset;
    } else if (axis == 3) {
      for (Layer input : inputs) {
        Halide::RDom r(0, input.w);
        forward(i, j, k, offset + r) = input.forward(i, j, k, r);
        offset += input.w;
      }
      w = offset;
    } else {
      throw new std::invalid_argument("Invalid axis dimension for concatenation");
    }
  }
};

class MVN : public Layer {
public:
  MVN(Layer input, Halide::Func mean, Halide::Func variance, float epsilon=1e-8)
    : Layer(input.x, input.y, input.z, input.w) {
    // TODO eventually compute this dynamically from a dataset
    forward(i, j, k, l) = (input.forward(i, j, k, l) - mean(i, j, k)) / (variance(i, j, k) + epsilon);
  }
};

} // namespace Espresso

#endif // ESPRESSO_COMPONENTS_H
