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

  Halide::Var cpu_schedule_embarrassingly_parallel(Halide::Func f, int tile_size=16, int vec_size=4) {
    Halide::Var i_inner, j_inner, vector_index, tile_index;
    f.tile(i, j, i_inner, j_inner, tile_size, tile_size)
     .vectorize(i_inner, vec_size)
     .fuse(i_inner, j_inner, vector_index)
     .unroll(vector_index, 4)
     .fuse(i, j, tile_index)
     .fuse(tile_index, k, tile_index)
     .fuse(tile_index, l, tile_index)
     .parallel(tile_index)
     .compute_root();

    return tile_index;
  }

  Halide::Var cpu_schedule_vision(Halide::Func f, int tile_size=16, int vec_size=4) {
    return cpu_schedule_embarrassingly_parallel(f, tile_size, vec_size);
  }

};

}

#endif // ESPRESSO_LAYER_H
