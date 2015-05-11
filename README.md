# espresso
Speeding up and debittering Caffe by adding Halide

[Website](http://jczhang.github.io/espresso/)

[Proposal](http://jczhang.github.io/espresso/proposal.html)

[Progress report](http://jczhang.github.io/espresso/checkpoint.html)

## Build instructions
Install [glog](https://code.google.com/p/google-glog/), [protobuf](https://github.com/google/protobuf), [Halide](https://github.com/halide/Halide/releases), then make. clang recommended.

- GTX 770 with OpenCL: 52.9 ms/image (30 batch)
- 4770K: 669 ms/image (18 batch)