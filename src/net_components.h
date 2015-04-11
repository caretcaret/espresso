#ifndef ESPRESSO_NET_COMPONENTS_H
#define ESPRESSO_NET_COMPONENTS_H

#include <iostream>
#include "Halide.h"

namespace Espresso {

/* input: the previous layer <n_input>
 * W: the weight matrix <n_input by n_output>
 * b: the bias vector <n_output>
 * output: this layer <n_output>
 */
Halide::Func hidden_layer(Halide::Func input, Halide::Func W, Halide::Func b, int n_input, int n_output) {
    Halide::Func output("output");
    Halide::Var i;
    Halide::RDom r(0, n_input);

    output(i) = Halide::max(Halide::sum(W(i, r.x) * input(r.x)) + b(i), 0);
    return output;
}

}

#endif // ESPRESSO_NET_COMPONENTS_H