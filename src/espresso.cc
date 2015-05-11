#include "espresso.h"
#include "test_espresso.h"
#include <vector>
#include <iostream>

int main(int argc, char **argv) {
  if (argc > 0) {
    google::InitGoogleLogging(argv[0]);
  }

 	std::vector<std::string> pics;

 	for (int i = 1; i < argc; i++) {
 		pics.push_back(std::string(argv[i]));
 	}

 	if (pics.size() == 0) {
 		std::cout << "No images specified.\n";
 		return -1;
 	}
  Espresso::test_main(pics);

  return 0;
}
