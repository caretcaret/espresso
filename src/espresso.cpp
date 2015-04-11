#include "Halide.h"

#include <iostream>
#include <random>

#include "image_util.h"
#include "net_components.h"


int main(int argc, char **argv) {
  std::random_device rd;
  std::default_random_engine generator(rd());

  int n_input = 5;
  int n_output = 7;

  // construct abstract network
  Halide::ImageParam W(Halide::type_of<double>(), 2);
  Halide::ImageParam b(Halide::type_of<double>(), 1);
  Halide::ImageParam input(Halide::type_of<double>(), 1);
  Halide::Func layer = Espresso::hidden_layer(Halide::Func(input), Halide::Func(W), Halide::Func(b), n_input, n_output);

  // instantiate inputs
  Halide::Image<double> W_(n_output, n_input);
  Halide::Image<double> b_(n_output);
  Halide::Image<double> input_(n_input);
  Espresso::fill_random(W_, generator, 0.0, 1.0);
  Espresso::fill_random(b_, generator, 0.0, 1.0);
  Espresso::fill_random(input_, generator, 0.0, 1.0);
  std::cout << "W:" << std::endl << W_ << std::endl;
  std::cout << "b:" << std::endl << b_ << std::endl;
  std::cout << "input:" << std::endl << input_ << std::endl;

  W.set(W_);
  b.set(b_);
  input.set(input_);

  // JIT compile and run
  Halide::Image<double> output = layer.realize(n_output);

  std::cout << "output:" << std::endl << output << std::endl;

  return 0;
}
