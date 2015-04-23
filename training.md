Rambling about how training may work:
The data is an input image of shape (# instances, feature dim1, feature dim2) and a target image of shape (# instances, output dim1, output dim2).
The input of training is an initial tuple of parameters.
The final output of training is a tuple of parameters that have been fit to the data.
To model the change in parameters over time, each parameter has a time dimension, and each instance is defined in terms of the previous. For each parameter p, p(t) is defined in terms of p(t-1), where the provided initial parameters is p(0) and the output parameters is p(t_f), where t_f is the number of epochs to run the training. This requires a static number of epochs to be specified at training init time, but if one wants a dynamic way to train, one can run some number of epochs at a time, and after each hyperepoch, check the error to see if another hyperepoch needs to be run. This is to prevent parameters from being copied between device and host repeatedly.
Specifying all parameters this way may seem inefficient, but should be made efficient with scheduling on a sliding window.
The network is a pipeline that is used as a component in training. It accepts a batch of input data of shape (# instances, feature dim1, feature dim2).
The network is built from a composition of layers. Each layer defines the transform it makes based on a vector of parameters and how to backpropagate results.
The transform takes in (# instances, feature dim1, feature dim2) with parameters and outputs (# instances, output dim1, output dim2). The programmer is responsible for making sure the dimensions match up; maybe we'll have some error checking mechanism when the layers are hooked up.
The layer is also responsible for specifying backprop derivative stuff:
for gradient descent, we need to update each parameter with param - alpha * d loss / d param. Chain rule: d/dx f(g(x)) = f'(g(x)) g'(x). Each layer fulfills the role of both f and g, so since it needs to implement df/dg * dg/dp, where df/dg = next layer's derivative, dg/dp = derivative of the current layer with respect to a parameter, each layer needs an implementation of df/dx and df/dp.
In doing so, each layer maintains a buffer for its parameters.
We can encode the loss function as a layer without params, and have the differentiation just work.

With these things, it *should* be possible to train a whole network with one kernel call (provided the amount of memory available to the gpu is sufficient).
