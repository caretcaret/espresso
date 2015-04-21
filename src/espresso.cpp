#include "Halide.h"

#include <iostream>
#include <random>

#include "image_util.h"
#include "net_components.h"


int main(int argc, char **argv) {
  std::random_device rd;
  std::default_random_engine generator(rd());

  int n_input = 5;
  int n_hidden = 7;
  int n_classes = 3;

  // construct abstract network
  Halide::ImageParam input(Halide::type_of<float>(), 4);
  Halide::ImageParam W1(Halide::type_of<float>(), 2);
  Halide::ImageParam b1(Halide::type_of<float>(), 1);
  Halide::ImageParam W2(Halide::type_of<float>(), 2);
  Halide::ImageParam b2(Halide::type_of<float>(), 1);

  Espresso::Component input_layer = Espresso::InputLayer(input, n_input, 1, 1, 1);
  Espresso::Component layer1 = Espresso::ReLU(Espresso::InnerProduct(input_layer, Halide::Func(W1), Halide::Func(b1), n_hidden));
  Espresso::Component layer2 = Espresso::Softmax(Espresso::InnerProduct(layer1, Halide::Func(W2), Halide::Func(b2), n_classes));

  // instantiate inputs
  Halide::Image<float> input_(n_input, 1, 1, 1);
  Halide::Image<float> W1_(n_hidden, n_input), W2_(n_classes, n_hidden);
  Halide::Image<float> b1_(n_hidden), b2_(n_classes);
  Espresso::fill_random(input_, generator, 0.0f, 1.0f);
  Espresso::fill_random(W1_, generator, 0.0f, 1.0f);
  Espresso::fill_random(b1_, generator, 0.0f, 1.0f);
  Espresso::fill_random(W2_, generator, 0.0f, 1.0f);
  Espresso::fill_random(b2_, generator, 0.0f, 1.0f);

  input.set(input_);
  W1.set(W1_);
  b1.set(b1_);
  W2.set(W2_);
  b2.set(b2_);

  // JIT compile and run
  Halide::Image<float> output = layer2.forward.realize(n_classes, 1, 1, 1);

  std::cout << "output:" << std::endl << output << std::endl;

  return 0;
}
