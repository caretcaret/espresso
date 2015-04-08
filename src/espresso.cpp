#include "Halide.h"

#include <iostream>
#include <random>


/* input: the previous layer <input_n>
 * W: the weight matrix <input_n by output_n>
 * b: the bias vector <output_n>
 * output: this layer <output_n>
 */
template<class RealType = double>
Halide::Func add_eval_layer(Halide::Image<RealType> input, Halide::Image<RealType> W, Halide::Image<RealType> b, int input_n, int output_n) {
  Halide::Func output("output");
  Halide::Var i;
  Halide::RDom r(0, input_n);

  output(i) = Halide::max(Halide::sum(W(i, r.x) * input(r.x)) + b(i), 0);

  return output;
}

/* Generate a random 2d matrix. */
template<class RealType = double>
Halide::Image<RealType> random_matrix(std::default_random_engine& generator, RealType mean, RealType stddev, int dimX, int dimY) {
  Halide::Image<RealType> M(dimX, dimY);
  std::normal_distribution<RealType> distribution(mean, stddev);
  for (int x = 0; x < dimX; x++) {
    for (int y = 0; y < dimY; y++) {
      M(x, y) = distribution(generator);
    }
  }

  return M;
}

/* Generate a random 1d vector. */
template<class RealType = double>
Halide::Image<RealType> random_vector(std::default_random_engine& generator, RealType mean, RealType stddev, int dimX) {
  Halide::Image<RealType> M(dimX);
  std::normal_distribution<RealType> distribution(mean, stddev);
  for (int x = 0; x < dimX; x++) {
      M(x) = distribution(generator);
  }

  return M;
}

template<class RealType>
void print_matrix(Halide::Image<RealType> M) {
  for (int i = 0; i < M.width(); i++) {
    for (int j = 0; j < M.height(); j++) {
      std::cout << M(i, j) << " ";
    }
    std::cout << std::endl;
  }
}

template<class RealType>
void print_vector(Halide::Image<RealType> v) {
  for (int i = 0; i < v.width(); i++) {
    std::cout << v(i) << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  std::random_device rd;
  std::default_random_engine generator(rd());

  int input_n = 5;
  int output_n = 7;

  Halide::Image<double> W = random_matrix(generator, 0.0, 1.0, output_n, input_n);
  Halide::Image<double> b = random_vector(generator, 0.0, 1.0, output_n);
  Halide::Image<double> input = random_vector(generator, 0.0, 1.0, input_n);

  std::cout << "W:" << std::endl;
  print_matrix(W);

  std::cout << "b:" << std::endl;
  print_vector(b);

  std::cout << "Input:" << std::endl;
  print_vector(input);

  Halide::Func layer = add_eval_layer(input, W, b, input_n, output_n);

  Halide::Image<double> output = layer.realize(output_n);

  std::cout << "Output:" << std::endl;
  print_vector(output);

  std::cout << "Done." << std::endl;

  return 0;
}
