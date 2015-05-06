#include <random>

#include "espresso.h"
#include "test_espresso.h"
#include "CycleTimer.h"

#include <glog/logging.h>

double run_net(Espresso::Layer output_layer) {
  // JIT compile before timing code
  output_layer.forward.compile_jit();

  double start_time = CycleTimer::currentSeconds();

  // run compiled code
  Halide::Image<float> output = output_layer.forward.realize(output_layer.x, output_layer.y, output_layer.z, output_layer.w);

  double end_time = CycleTimer::currentSeconds();

  LOG(INFO) << output;
  LOG(INFO) << end_time - start_time;

  return end_time - start_time;
}

double test_convolution(std::default_random_engine& generator,
    int input_x=16, int input_y=32, int input_z=4, int input_w=2, int kernel_x=5, int kernel_y=5, int n_filters=8,
    int pad_x=2, int pad_y=2, int stride_x=3, int stride_y=3, bool bias_term=true, int group=2) {
  Halide::ImageParam input(Halide::type_of<float>(), 4);
  Halide::ImageParam kernel(Halide::type_of<float>(), 4);

  Espresso::Layer input_layer = Espresso::MemoryData(input, input_x, input_y, input_z, input_w);
  Espresso::Layer output_layer = Espresso::Convolution(input_layer, Halide::Func(kernel),
    kernel_x, kernel_y, n_filters, pad_x, pad_y, stride_x, stride_y, bias_term, group);

  // instantiate inputs
  Halide::Image<float> input_(input_x, input_y, input_z, input_w);
  Halide::Image<float> kernel_(kernel_x, kernel_y, input_z+1, n_filters);
  Espresso::fill_random(input_, generator, 0.0f, 1.0f);
  Espresso::fill_random(kernel_, generator, 0.0f, 1.0f);

  LOG(INFO) << input_;
  LOG(INFO) << kernel_;

  input.set(input_);
  kernel.set(kernel_);

  return run_net(output_layer);
}

double test_pooling(std::default_random_engine& generator,
    std::string method="max", int input_x=16, int input_y=32, int input_z=3, int input_w=2, int pool_x=5, int pool_y=5,
    int pad_x=0, int pad_y=0, int stride_x=1, int stride_y=1) {
  Halide::ImageParam input(Halide::type_of<float>(), 4);

  Espresso::Layer input_layer = Espresso::MemoryData(input, input_x, input_y, input_z, input_w);
  Espresso::Layer output_layer = Espresso::Pooling(input_layer, method, pool_x, pool_y, pad_x, pad_y, stride_x, stride_y);

  Halide::Image<float> input_(input_x, input_y, input_z, input_w);
  Espresso::fill_random(input_, generator, 0.0f, 1.0f);

  LOG(INFO) << input_;

  input.set(input_);

  return run_net(output_layer);
}

double test_LRN(std::default_random_engine& generator,
    int input_x=16, int input_y=32, int input_z=3, int input_w=2, int region_x=1, int region_y=1, int region_z=3,
    float alpha=1.0f, float beta=5.0f) {
  Halide::ImageParam input(Halide::type_of<float>(), 4);

  Espresso::Layer input_layer = Espresso::MemoryData(input, input_x, input_y, input_z, input_w);
  Espresso::Layer output_layer = Espresso::LRN(input_layer, region_x, region_y, region_z, alpha, beta);

  Halide::Image<float> input_(input_x, input_y, input_z, input_w);
  Espresso::fill_random(input_, generator, 0.0f, 1.0f);

  LOG(INFO) << input_;

  input.set(input_);

  return run_net(output_layer);
}

double test_random(std::default_random_engine& generator,
    int n_input=5, int n_hidden=7, int n_classes=3) {
  // construct abstract network
  Halide::ImageParam input(Halide::type_of<float>(), 4);
  Halide::ImageParam W1(Halide::type_of<float>(), 2);
  Halide::ImageParam W2(Halide::type_of<float>(), 2);

  Espresso::Layer input_layer = Espresso::MemoryData(input, n_input, 1, 1, 1);
  Espresso::Layer layer1 = Espresso::ReLU(Espresso::InnerProduct(input_layer, Halide::Func(W1), n_hidden));
  Espresso::Layer layer2 = Espresso::Softmax(Espresso::InnerProduct(layer1, Halide::Func(W2), n_classes));

  // instantiate inputs
  Halide::Image<float> input_(n_input, 1, 1, 1);
  Halide::Image<float> W1_(n_hidden, n_input + 1), W2_(n_classes, n_hidden + 1);
  Espresso::fill_random(input_, generator, 0.0f, 1.0f);
  Espresso::fill_random(W1_, generator, 0.0f, 1.0f);
  Espresso::fill_random(W2_, generator, 0.0f, 1.0f);

  input.set(input_);
  W1.set(W1_);
  W2.set(W2_);

  LOG(INFO) << input_;
  LOG(INFO) << W1_;
  LOG(INFO) << W2_;

  return run_net(layer2);
}

namespace Espresso {

int test_main() {
  std::random_device rd;
  std::default_random_engine generator(rd());

  test_random(generator);
  test_convolution(generator);
  test_pooling(generator);
  test_LRN(generator);
  return 0;
}

} // namespace Espresso
