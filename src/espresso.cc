#include <iostream>

#include "espresso.h"
#include "test_espresso.h"


int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  Espresso::SolverParameter solve_params;
  Espresso::NetParameter net_params;

  ReadSolverParamsFromTextFile("./models/bvlc_reference_caffenet/solver.prototxt", &solve_params);
  ReadNetParamsFromTextFile(solve_params.net(), &net_params);

  std::cout << net_params.name() << std::endl;

  Espresso::test_main();

  return 0;
}
