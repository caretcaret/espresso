#ifndef ESPRESSO_LAYER_H
#define ESPRESSO_LAYER_H

#include "Halide.h"

namespace Espresso {
/**************/
/* Base class */
/**************/

class Layer {
public:
  Halide::Func forward;
  /* Dimensions of the layer output. Operations on
   * fewer than 4 dimensions perform by batching up
   * dimensions w, z, and/or y. */
  int x, y, z, w;

  Layer(int x, int y, int z, int w)
    : forward("forward"), x(x), y(y), z(z), w(w), i("i"), j("j"), k("k"), l("l") {}

protected:
  /* Variables that are used everywhere */
  Halide::Var i, j, k, l;
  std::vector<std::shared_ptr<Halide::Func> > parameters;
  // used to denote uninitialized layer
  Layer() : Layer(0, 0, 0, 0) {}
};

}

#endif // ESPRESSO_LAYER_H
