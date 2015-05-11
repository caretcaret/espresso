# Espresso
### Speeding up and debittering Caffe by adding Halide

## Summary
Espresso is a partial implementation of Caffe in the Halide image processing language. It supports fine-tuning of deep neural networks to achieve high evaluation performance on multiple platforms.

## Background

### Caffe
Caffe is a deep learning framework developed by the Berkeley Vision and Learning Center. It features GPU evaluation and training of convolutional neural networks with a clean interface. Models are described by configuration files and trained models can be exported.

### Halide
Halide is an domain-specific language embedded in `C++`. It decouples algorithm implementations from scheduling and organization of computation, making it easier to optimize image processing pipelines for performance on multiple platforms.

### CaffeNet
CaffeNet is a 9-layer DNN trained on ILSVRC 2012. It takes `256x256x3` images as input, and outputs a 1000-wide probability vector, where every item in the vector corresponds with a class. The first five layers are convolutional, and the next three are fully-connected. The last fully-connected layer is connected with a softmax which produces the distribution over the 1000 class labels.

## Approach

### Architecture
DNNs in Espresso are expressed as compositions of `[Layer]`(https://github.com/jczhang/espresso/include/layer.h)s. Every `Layer` takes a `Layer` as input, besides the `MemoryData Layer`, which is used for input. We implemented most of the layers found in Caffe. Input is represented as a four-dimensional matrix `(x, y, z, n)`, where `(x, y)` is a single image, `z` represents the channels in a single image, and `n` is the image index. For our CaffeNet demo, we take all images in the PPM format from an input directory. By expressing input in this way, we can run multiple images through CaffeNet simultaneously, ensuring that we maximize the amount of work done on the GPU before copying data back out to the CPU. DNNs can be loaded from a `.caffemodel` file, or built from hand in `C++`. Each individual layer, as a Halide construct, consists of an algorithm and a schedule. Specifically, every `Layer` has a `Halide::Func forward`, which represents the output of the layer. (Currently, Espresso does not support back propogation for training). These functions are defined in the same four dimensions as the input, though they are not constrained in what the dimension sizes actually are.

### Scheduling Layers
Scheduling in Halide defines order and locality, and allows us to exploit CPU or GPU parallelism. In Espresso, we aggressively schedule layers to achieve maximum performance. We focused on GPU performance, so our scheduling is done solely through Halide's GPU scheduling primitives. There are a few patterns in scheduling that we found to be very effective. First, tiling pieces of layers and assigning them to thread blocks dramatically sped up computation in most cases, as we were able to exploit both locality and warps not diverging. This pattern works best when the layer requires surrounding pixels to compute, such as in a convolution. Another effective pattern was a call to `vectorize()`, which is essentially a one-dimensional tile. This pattern works best on layers with only horizontal memory accesses, like our `InnerProduct` layer. We have provided helper functions for both of these patterns.

Halide aggressively inlines functions in multi-stage pipelines. What this means for Espresso is that if a function is not explicitly scheduled, it will most likely be inlined into the next explictly scheduled step in the Espresso pipeline. This can improve performance and memory usage, since we do not have to store an intermediate image for the inlined function.

## Results
We tested our implementation of CaffeNet in Espresso using an nVIDIA GTX 770 GPU with 2GB of RAM. Our reference benchmark for both performance and correctness was the same network run in Caffe.

gtx770, batch size 30
fastish
profile graph
future work - potential improvements, time

## References

## Work
Work was divided evenly.
