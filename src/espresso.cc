#include "espresso.h"
#include "test_espresso.h"


int main(int argc, char **argv) {
  if (argc > 0) {
    google::InitGoogleLogging(argv[0]);
  }

  Espresso::test_main("./images/test/cat.ppm");

  return 0;
}
