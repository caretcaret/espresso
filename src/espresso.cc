#include <iostream>

#include "espresso.h"
#include "test_espresso.h"


int main(int argc, char **argv) {
  if (argc > 0) {
    google::InitGoogleLogging(argv[0]);
  }

  Espresso::Net net("./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel");
  std::cout << net.name << std::endl;

  Espresso::test_main();

  return 0;
}
