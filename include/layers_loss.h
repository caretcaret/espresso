#ifndef ESPRESSO_LAYERS_LOSS_H
#define ESPRESSO_LAYERS_LOSS_H

#include "Halide.h"
#include "layer.h"

namespace Espresso {

/***************/
/* Loss layers */
/***************/

class MultinomialLogisticLoss : public Layer {
public:
  MultinomialLogisticLoss(Layer pred, Layer labels)
    : Layer(1, 1, 1, 1) {
    // predictions are probability vectors num_classes by 1 by 1 by batch_size
    // labels are integers 1 by 1 by 1 by batch_size
    Halide::RDom r(0, pred.w);

    forward(i, j, k, l) = -Halide::sum(Halide::fast_log(pred.forward(labels.forward(0, 0, 0, r.x), 0, 0, r.x))) / pred.w;
  }
};

class EuclideanLoss : public Layer {
public:
  EuclideanLoss(Layer pred, Layer obs)
    : Layer(1, 1, 1, 1) {
    // predictions and observations are 1 by 1 by 1 by batch_size
    Halide::Func diff;
    Halide::RDom r(0, pred.w);

    diff(w) = pred.forward(0, 0, 0, w) - obs.forward(0, 0, 0, w);
    forward(i, j, k, l) = Halide::sum(diff(r.x) * diff(r.x)) / (2 * pred.w);
  }
};

class InfoGainLoss : public Layer {
public:
  InfoGainLoss(Layer pred, Layer labels, Halide::Func infogain)
    : Layer(1, 1, 1, 1) {
    // infogain is a n_class by n_class matrix
    Halide::RDom r(0, pred.x, 0, pred.w);
    Halide::RVar cls = r.x;
    Halide::RVar w = r.y;

    forward(i, j, k, l) = -Halide::sum(infogain(labels.forward(0, 0, 0, w), cls) * Halide::fast_log(pred.forward(cls, 0, 0, w))) / pred.w;
  }
};

class Accuracy : public Layer {
public:
  Accuracy(Layer pred, Layer labels, int top_k=1)
    : Layer(1, 1, 1, 1) {
    Halide::Func correct;
    Halide::RDom r(0, pred.x);
    Halide::RDom s(0, pred.w);

    if (top_k != 1) {
      // TODO
      throw new std::invalid_argument("Top-k not implemented for k != 1");
    }

    correct(l) = Halide::argmax(r, pred.forward(r.x, 0, 0, l))[0] == labels.forward(0, 0, 0, l);
    forward(i, j, k, l) = Halide::sum(Halide::select(correct(s.x), 1, 0)) / pred.w;
  }
};

class HingeLoss : public Layer {
public:
  HingeLoss(Layer pred, Layer labels, int p)
    : Layer(1, 1, 1, 1) {
    Halide::Func loss;
    Halide::RDom r(0, pred.x, 0, pred.w);

    loss(i, l) = Halide::max(0, 1 + Halide::select(labels.forward(0, 0, 0, l) == i, -1, 1) * pred.forward(i, 0, 0, l));
    forward(i, j, k, l) = Halide::sum(Halide::fast_pow(loss(r.x, r.y), p)) / pred.w;
  }
};

} // namespace Espresso

#endif // ESPRESSO_LAYERS_LOSS_H
