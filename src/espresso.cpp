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
  Halide::ImageParam input(Halide::type_of<double>(), 1);
  Halide::ImageParam W1(Halide::type_of<double>(), 2);
  Halide::ImageParam b1(Halide::type_of<double>(), 1);
  Halide::ImageParam W2(Halide::type_of<double>(), 2);
  Halide::ImageParam b2(Halide::type_of<double>(), 1);

  Halide::Func layer1 = Espresso::hidden_layer(Halide::Func(input), Halide::Func(W1), Halide::Func(b1), n_input, n_hidden);
  Halide::Func layer2 = Espresso::softmax(layer1, Halide::Func(W2), Halide::Func(b2), n_hidden, n_classes);

  // instantiate inputs
  Halide::Image<double> input_(n_input);
  Halide::Image<double> W1_(n_hidden, n_input), W2_(n_classes, n_hidden);
  Halide::Image<double> b1_(n_hidden), b2_(n_classes);
  Espresso::fill_random(input_, generator, 0.0, 1.0);
  Espresso::fill_random(W1_, generator, 0.0, 1.0);
  Espresso::fill_random(b1_, generator, 0.0, 1.0);
  Espresso::fill_random(W2_, generator, 0.0, 1.0);
  Espresso::fill_random(b2_, generator, 0.0, 1.0);

  input.set(input_);
  W1.set(W1_);
  b1.set(b1_);
  W2.set(W2_);
  b2.set(b2_);

  // JIT compile and run
  Halide::Image<double> output = layer2.realize(n_classes);

  std::cout << "output:" << std::endl << output << std::endl;

  return 0;
}
