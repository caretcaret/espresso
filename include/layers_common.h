#ifndef ESPRESSO_LAYERS_COMMON_H
#define ESPRESSO_LAYERS_COMMON_H

#include "Halide.h"
#include "layer.h"

namespace Espresso {

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
  /* W: output_x * input.x * 1 * 1
   * bias: output_x * 1 * 1 * 1 */
  /* TODO: use blas gemv/gemm */
  InnerProduct(Layer input, Halide::Func W, Halide::Func bias, int output_x, bool bias_term=true)
    : Layer(output_x, input.y, input.z, input.w) {
    Halide::RDom r(0, input.x);

    if (bias_term) {
      forward(i, j, k, l) = Halide::sum(W(r.x, i, 0, 0) * input.forward(r.x, j, k, l)) + bias(i, 0, 0, 0);
    } else {
      forward(i, j, k, l) = Halide::sum(W(r.x, i, 0, 0) * input.forward(r.x, j, k, l));
    }

    forward.vectorize(i, 256).parallel(k).parallel(l).compute_root();
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
  Eltwise(std::string op, Layer input1, Layer input2)
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

    forward.gpu_tile(i, j, 16, 16);
    forward.compute_root();

  }
};

class Concat : public Layer {
public:
  Concat(int axis, std::initializer_list<Layer> inputs)
    // not the actual dimensions; the axis of concatenation is set again below
    : Layer(inputs.begin()->x, inputs.begin()->y, inputs.begin()->z, inputs.begin()->w) {
    // layers may only be concatenated along one axis, all other axes must have the same dimensions

    int offset = 0;
    forward(i, j, k, l) = 0.0f;

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

#endif // ESPRESSO_LAYERS_COMMON_H
