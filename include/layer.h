#ifndef ESPRESSO_LAYER_H
#define ESPRESSO_LAYER_H

#include "Halide.h"
#include <map>

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
  
protected:
  /* Variables that are used everywhere */
  Halide::Var i, j, k, l;
  std::map<int, Halide::Image<float> > parameters;

  /* Default constructor */
  Layer(int x=0, int y=0, int z=0, int w=0)
    : forward("forward"), x(x), y(y), z(z), w(w), i("i"), j("j"), k("k"), l("l") {}

  void set_dim(int x, int y, int z, int w) {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
  }
};


}

#endif // ESPRESSO_LAYER_H
