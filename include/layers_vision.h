#ifndef ESPRESSO_COMPONENTS_H
#define ESPRESSO_COMPONENTS_H

#include "Halide.h"
#include "layer.h"
#include "image_util.h"
#include "proto/caffe.pb.h"

namespace Espresso {

/*****************/
/* Vision layers */
/*****************/

class Convolution : public Layer {
public:
  Convolution(Layer input, Halide::Func kernel, Halide::Func bias, int kernel_x, int kernel_y, int n_filters,
      int pad_x=0, int pad_y=0, int stride_x=1, int stride_y=1, bool bias_term=true, int group=1)
    : Layer((input.x + 2 * pad_x - kernel_x) / stride_x + 1,
            (input.y + 2 * pad_y - kernel_y) / stride_y + 1,
            n_filters,
            input.w) {
    // Kernel size is kernel_x by kernel_y by input.z / group by n_filters, where n_filters is the number of filters
    // Bias size is n_filters by 1 by 1 by 1
    // kernel_x, kernel_y must be odd
    int input_group_size = input.z / group;
    int output_group_size = n_filters / group;
    Halide::Func clamped = Halide::BoundaryConditions::constant_exterior(input.forward, 0.0f, 0, input.x, 0, input.y);
    Halide::Func padded("padded"), convolved("convolved");
    Halide::RDom r(-kernel_x / 2, kernel_x / 2 + 1, -kernel_y / 2, kernel_y / 2 + 1, 0, input_group_size);
    Halide::Expr group_num = k / output_group_size, group_idx = k % output_group_size;

    padded(i, j, k, l) = Halide::select(
        (i % (2 * pad_x + 1) == 0) && (j % (2 * pad_y + 1) == 0),
        clamped(i / (2 * pad_x + 1), j / (2 * pad_y + 1), k, l),
        0.0f);

    if (bias_term) {
      convolved(i, j, k, l) = Halide::sum(padded(i + r.x, j + r.y, group_num * input_group_size + r.z, l) *
          kernel(r.x + kernel_x / 2, r.y + kernel_y / 2, r.z, group_num * output_group_size + group_idx)) + bias(k, 0, 0, 0);
    } else {
      convolved(i, j, k, l) = Halide::sum(padded(i + r.x, j + r.y, group_num * input_group_size + r.z, l) *
          kernel(r.x + kernel_x / 2, r.y + kernel_y / 2, r.z, group_num * output_group_size + group_idx));
    }

    forward(i, j, k, l) = convolved(i * stride_x, j * stride_y, k, l);

    convolved.compute_root();
    // forward.gpu_tile(i, j, 16, 16).compute_root();
  }

  Convolution(const LayerParameter& param) : Layer() {
    Halide::Image<float> kernel_ = Espresso::from_blob(param.blobs(0));
    Halide::Image<float> bias_ = Espresso::from_blob(param.blobs(1));
    Halide::Func kernel(kernel_);
    Halide::Func bias(bias_);

    // TODO move constructor body into init, call init with options
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
    Halide::Func clamped = Halide::BoundaryConditions::constant_exterior(input.forward, 0.0f, 0, input.x, 0, input.y);
    Halide::Func padded("padded");
    Halide::Func pooled("pooled");
    Halide::Func rand_x, rand_y;
    Halide::RDom r(-pool_x / 2, pool_x / 2 + 1, -pool_y / 2, pool_y / 2 + 1);

    padded(i, j, k, l) = Halide::select(
        (i % (2 * pad_x + 1) == 0) && (j % (2 * pad_y + 1) == 0),
        clamped(i / (2 * pad_x + 1), j / (2 * pad_y + 1), k, l),
        0.0f);

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

    forward.gpu_tile(i, j, 16, 16).compute_root();
  }
};

class LRN : public Layer {
public:
  LRN(Layer input, int region_x=1, int region_y=1, int region_z=1, float alpha=1.0f, float beta=5.0f)
    : Layer(input.x, input.y, input.z, input.w) {
    // across_channels => region_x and region_y = 1; within_channels => region_z = 1
    Halide::Func clamped = Halide::BoundaryConditions::constant_exterior(input.forward, 0.0f, 0, input.x, 0, input.y, 0, input.z);
    Halide::Func activation("activation");
    Halide::Func normalizer("normalizer");
    Halide::RDom r(-region_x / 2, region_x / 2 + 1, -region_y / 2, region_y / 2 + 1, -region_z / 2, region_z / 2 + 1);

    activation(i, j, k, l) = Halide::sum(clamped(i + r.x, j + r.y, k + r.z, l));
    normalizer(i, j, k, l) = Halide::fast_pow(1 + (alpha / (region_x * region_y * region_z)) * activation(i, j, k, l), beta);
    forward(i, j, k, l) = activation(i, j, k, l) / normalizer(i, j, k, l);

    // Halide::Var i_inner, i_outer, j_inner, j_outer, tile_index;

    // Halide::Var i_inner2, i_outer2, j_inner2, j_outer2, tile_index2;
    // forward.tile(i, j, i_outer2, j_outer2, i_inner2, j_inner2, 4, 4);
    // forward.unroll(i_inner2).unroll(j_inner2);
    // forward.fuse(i_outer2, j_outer2, tile_index2);
    // forward.parallel(tile_index2);
    // forward.compute_root();

    // activation.compute_at(forward, tile_index2);

    forward.gpu_tile(i, j, 16, 16);
    forward.compute_root();
  }
};

} // namespace Espresso

#endif // ESPRESSO_COMPONENTS_H
