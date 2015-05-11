#include <random>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include "espresso.h"
#include "CycleTimer.h"

#include <glog/logging.h>


namespace Espresso {

#define DEFAULT_N_IMAGES 10

double run_net(Espresso::Layer output_layer, bool use_gpu=false, int n_images=DEFAULT_N_IMAGES) {
  // JIT compile before timing code

  LOG(INFO) << "Compiling...";

  if (use_gpu) {
    Halide::Target target = Halide::get_host_target();
    target.set_feature(Halide::Target::OpenCL);

    output_layer.forward.compile_jit(target);
  }
  else {
    output_layer.forward.compile_jit();
  }

  double start_time = CycleTimer::currentSeconds();

  LOG(INFO) << "Running... ";

  // run compiled code and place it into a buffer (still on the GPU for timing purposes)
  Halide::Buffer output_buffer(Halide::Float(32), output_layer.x, output_layer.y, output_layer.z, output_layer.w);
  output_layer.forward.realize(output_buffer);

  double end_time = CycleTimer::currentSeconds();

  // Move the buffer onto the CPU
  Halide::Image<float> output(output_buffer);

  LOG(INFO) << (end_time - start_time) / n_images * 1000 << "ms/image";

  return end_time - start_time;
}

Halide::Image<float> run_net_with_output(Espresso::Layer output_layer, bool use_gpu=false, int n_images=DEFAULT_N_IMAGES) {
  // JIT compile before timing code

  std::cout << "Compiling...\n";

  if (use_gpu) {
    Halide::Target target = Halide::get_host_target();
    target.set_feature(Halide::Target::OpenCL);

    output_layer.forward.compile_jit(target);
  }
  else {
    output_layer.forward.compile_jit();
  }

  std::cout << "Running... ";
  std::cout.flush();

  double start_time = CycleTimer::currentSeconds();

  // run compiled code and place it into a buffer (still on the GPU for timing purposes)
  Halide::Buffer output_buffer(Halide::Float(32), output_layer.x, output_layer.y, output_layer.z, output_layer.w);
  output_layer.forward.realize(output_buffer);

  double end_time = CycleTimer::currentSeconds();

  // Move the buffer onto the CPU
  Halide::Image<float> output(output_buffer);
  LOG(INFO) << output;

  std::cout << (end_time - start_time) / n_images * 1000 << " ms/image";

  return output;
}

// yolohack lol
Layer bvlc_reference_caffenet(Halide::ImageParam input_data, Halide::ImageParam input_labels, int n_images=DEFAULT_N_IMAGES) {
  NetParameter param;
  ReadNetParamsFromBinaryFile("./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel", &param);

  std::cout << "Loading images...\n";

  Espresso::Layer data = Espresso::MemoryData(input_data, 227, 227, 3, n_images); //1
  Espresso::Layer labels = Espresso::MemoryData(input_labels, 1, 1, 1, n_images); //2

  int conv1_filters = 96, conv1_size = 11, conv1_stride = 4;
  Halide::ImageParam kernel1(Halide::type_of<float>(), 4);
  Halide::ImageParam bias1(Halide::type_of<float>(), 1);
  Halide::Image<float> kernel1_ = from_blob(param.layers(1).blobs(0));
  Halide::Image<float> bias1_ = from_blob(param.layers(1).blobs(1));
  kernel1.set(kernel1_);
  bias1.set(bias1_);
  Espresso::Layer conv1 = Espresso::Convolution(data, Halide::Func(kernel1), Halide::Func(bias1), conv1_size, conv1_size, conv1_filters, 0, 0, conv1_stride, conv1_stride); //3

  Espresso::Layer relu1 = Espresso::ReLU(conv1); //4

  int pool1_size = 3, pool1_stride = 2;
  Espresso::Layer pool1 = Espresso::Pooling(relu1, "max", pool1_size, pool1_size, 0, 0, pool1_stride, pool1_stride); //5

  int norm1_size = 5;
  float norm1_alpha = 0.0001f, norm1_beta = 0.75f;
  Espresso::Layer norm1 = Espresso::LRN(pool1, 1, 1, norm1_size, norm1_alpha, norm1_beta); //6

  int conv2_filters = 256, conv2_pad = 2, conv2_size = 5, conv2_group = 2;
  Halide::ImageParam kernel2(Halide::type_of<float>(), 4);
  Halide::ImageParam bias2(Halide::type_of<float>(), 1);
  Halide::Image<float> kernel2_ = from_blob(param.layers(5).blobs(0));
  Halide::Image<float> bias2_ = from_blob(param.layers(5).blobs(1));
  kernel2.set(kernel2_);
  bias2.set(bias2_);
  Espresso::Layer conv2 = Espresso::Convolution(norm1, Halide::Func(kernel2), Halide::Func(bias2), conv2_size, conv2_size, conv2_filters, conv2_pad, conv2_pad, 1, 1, true, conv2_group); //7

  Espresso::Layer relu2 = Espresso::ReLU(conv2); //8


  int pool2_size = 3, pool2_stride = 2;
  Espresso::Layer pool2 = Espresso::Pooling(relu2, "max", pool2_size, pool2_size, 0, 0, pool2_stride, pool2_stride); //9

  int norm2_size = 5;
  float norm2_alpha = 0.0001f, norm2_beta = 0.75f;
  Espresso::Layer norm2 = Espresso::LRN(pool2, 1, 1, norm2_size, norm2_alpha, norm2_beta); //10

  int conv3_filters = 384, conv3_pad = 1, conv3_size = 3;
  Halide::ImageParam kernel3(Halide::type_of<float>(), 4);
  Halide::ImageParam bias3(Halide::type_of<float>(), 1);
  Halide::Image<float> kernel3_ = from_blob(param.layers(9).blobs(0));
  Halide::Image<float> bias3_ = from_blob(param.layers(9).blobs(1));
  kernel3.set(kernel3_);
  bias3.set(bias3_);
  Espresso::Layer conv3 = Espresso::Convolution(norm2, Halide::Func(kernel3), Halide::Func(bias3), conv3_size, conv3_size, conv3_filters, conv3_pad, conv3_pad); //11

  Espresso::Layer relu3 = Espresso::ReLU(conv3); //12

  int conv4_filters = 384, conv4_pad = 1, conv4_size = 3, conv4_group = 2;
  Halide::ImageParam kernel4(Halide::type_of<float>(), 4);
  Halide::ImageParam bias4(Halide::type_of<float>(), 1);
  Halide::Image<float> kernel4_ = from_blob(param.layers(11).blobs(0));
  Halide::Image<float> bias4_ = from_blob(param.layers(11).blobs(1));
  kernel4.set(kernel4_);
  bias4.set(bias4_);
  Espresso::Layer conv4 = Espresso::Convolution(relu3, Halide::Func(kernel4), Halide::Func(bias4), conv4_size, conv4_size, conv4_filters, conv4_pad, conv4_pad, 1, 1, true, conv4_group); //13

  Espresso::Layer relu4 = Espresso::ReLU(conv4); //14

  int conv5_filters = 256, conv5_pad = 1, conv5_size = 3, conv5_group = 2;
  Halide::ImageParam kernel5(Halide::type_of<float>(), 4);
  Halide::ImageParam bias5(Halide::type_of<float>(), 1);
  Halide::Image<float> kernel5_ = from_blob(param.layers(13).blobs(0));
  Halide::Image<float> bias5_ = from_blob(param.layers(13).blobs(1));
  kernel5.set(kernel5_);
  bias5.set(bias5_);
  Espresso::Layer conv5 = Espresso::Convolution(relu4, Halide::Func(kernel5), Halide::Func(bias5), conv5_size, conv5_size, conv5_filters, conv5_pad, conv5_pad, 1, 1, true, conv5_group); //15

  Espresso::Layer relu5 = Espresso::ReLU(conv5); //16

  int pool5_size = 3, pool5_stride = 2;
  Espresso::Layer pool5 = Espresso::Pooling(relu5, "max", pool5_size, pool5_size, 0, 0, pool5_stride, pool5_stride); //17

  // TODO: revisit InnerProduct to implicitly flatten
  Espresso::Layer flatten5 = Espresso::Flatten(pool5); //18

  int fc6_size = 4096;
  Halide::ImageParam W6(Halide::type_of<float>(), 2);
  Halide::ImageParam bias6(Halide::type_of<float>(), 1);
  Halide::Image<float> W6_ = transpose(from_blob(param.layers(16).blobs(0)));
  Halide::Image<float> bias6_ = from_blob(param.layers(16).blobs(1));
  W6.set(W6_);
  bias6.set(bias6_);
  Espresso::Layer fc6 = Espresso::InnerProduct(flatten5, Halide::Func(W6), Halide::Func(bias6), fc6_size); //19

  Espresso::Layer relu6 = Espresso::ReLU(fc6); //20

  Espresso::Layer drop6 = Espresso::Dropout(relu6); //21

  int fc7_size = 4096;
  Halide::ImageParam W7(Halide::type_of<float>(), 2);
  Halide::ImageParam bias7(Halide::type_of<float>(), 1);
  Halide::Image<float> W7_ = transpose(from_blob(param.layers(19).blobs(0)));
  Halide::Image<float> bias7_ = from_blob(param.layers(19).blobs(1));
  W7.set(W7_);
  bias7.set(bias7_);
  Espresso::Layer fc7 = Espresso::InnerProduct(drop6, Halide::Func(W7), Halide::Func(bias7), fc7_size); //22

  Espresso::Layer relu7 = Espresso::ReLU(fc7); //23

  Espresso::Layer drop7 = Espresso::Dropout(relu7); //24

  int fc8_size = 1000;
  Halide::ImageParam W8(Halide::type_of<float>(), 2);
  Halide::ImageParam bias8(Halide::type_of<float>(), 1);
  Halide::Image<float> W8_ = transpose(from_blob(param.layers(22).blobs(0)));
  Halide::Image<float> bias8_ = from_blob(param.layers(22).blobs(1));
  W8.set(W8_);
  bias8.set(bias8_);
  Espresso::Layer fc8 = Espresso::InnerProduct(drop7, Halide::Func(W8), Halide::Func(bias8), fc8_size); //25

  Espresso::Layer accuracy = Espresso::Accuracy(fc8, labels); //26

  Espresso::Layer softmax = Espresso::Softmax(fc8); //27
  Espresso::Layer loss = Espresso::MultinomialLogisticLoss(softmax, labels);  //28

  return softmax; // TODO: screw it not returning loss/accuracy atm

}

Halide::Image<float> test_convolution(std::default_random_engine& generator,
    int input_x=16, int input_y=16, int input_z=1, int input_w=1, int kernel_x=5, int kernel_y=5, int n_filters=1,
    int pad_x=2, int pad_y=2, int stride_x=3, int stride_y=3, bool bias_term=true, int group=1) {
  Halide::ImageParam input(Halide::type_of<float>(), 4);
  Halide::ImageParam kernel(Halide::type_of<float>(), 4);
  Halide::ImageParam bias(Halide::type_of<float>(), 1);

  Espresso::Layer input_layer = Espresso::MemoryData(input, input_x, input_y, input_z, input_w);
  Espresso::Layer output_layer = Espresso::Convolution(input_layer, Halide::Func(kernel), Halide::Func(bias),
    kernel_x, kernel_y, n_filters, pad_x, pad_y, stride_x, stride_y, bias_term, group);

  // instantiate inputs
  Halide::Image<float> input_(input_x, input_y, input_z, input_w);
  Halide::Image<float> kernel_(kernel_x, kernel_y, input_z, n_filters);
  Halide::Image<float> bias_(n_filters, 1, 1, 1);
  Espresso::fill_random(input_, generator, 0.0f, 1.0f);
  Espresso::fill_random(kernel_, generator, 0.0f, 1.0f);
  Espresso::fill_random(bias_, generator, 0.0f, 0.0f);

  // LOG(INFO) << input_;
  // LOG(INFO) << kernel_;

  input.set(input_);
  kernel.set(kernel_);
  bias.set(bias_);

  return run_net_with_output(output_layer, true);
}

double test_pooling(std::default_random_engine& generator,
    std::string method="max", int input_x=256, int input_y=256, int input_z=3, int input_w=2, int pool_x=5, int pool_y=5,
    int pad_x=0, int pad_y=0, int stride_x=1, int stride_y=1) {
  Halide::ImageParam input(Halide::type_of<float>(), 4);

  Espresso::Layer input_layer = Espresso::MemoryData(input, input_x, input_y, input_z, input_w);
  Espresso::Layer output_layer = Espresso::Pooling(input_layer, method, pool_x, pool_y, pad_x, pad_y, stride_x, stride_y);

  Halide::Image<float> input_(input_x, input_y, input_z, input_w);
  Espresso::fill_random(input_, generator, 0.0f, 1.0f);

  // LOG(INFO) << input_;

  input.set(input_);

  return run_net(output_layer, true);
}

double test_LRN(std::default_random_engine& generator,
    int input_x=16, int input_y=32, int input_z=3, int input_w=2, int region_x=1, int region_y=1, int region_z=3,
    float alpha=1.0f, float beta=5.0f) {
  Halide::ImageParam input(Halide::type_of<float>(), 4);

  Espresso::Layer input_layer = Espresso::MemoryData(input, input_x, input_y, input_z, input_w);
  Espresso::Layer output_layer = Espresso::LRN(input_layer, region_x, region_y, region_z, alpha, beta);

  Halide::Image<float> input_(input_x, input_y, input_z, input_w);
  Espresso::fill_random(input_, generator, 0.0f, 1.0f);

  // LOG(INFO) << input_;

  input.set(input_);

  return run_net(output_layer, true);
}

double test_flatten(std::default_random_engine& generator,
    int input_x=5, int input_y=7, int input_z=2, int input_w=3) {
  Halide::ImageParam input(Halide::type_of<float>(), 4);

  Espresso::Layer input_layer = Espresso::MemoryData(input, input_x, input_y, input_z, input_w);
  Espresso::Layer output_layer = Espresso::Flatten(input_layer);

  Halide::Image<float> input_(input_x, input_y, input_z, input_w);
  Espresso::fill_random(input_, generator, 0.0f, 1.0f);

  input.set(input_);

  // LOG(INFO) << input_;

  return run_net(output_layer, true);
}

double test_eltwise(std::default_random_engine& generator,
    int input_x=5, int input_y=7, int input_z=2, int input_w=3) {
  Halide::ImageParam input(Halide::type_of<float>(), 4);

  Espresso::Layer input_layer = Espresso::MemoryData(input, input_x, input_y, input_z, input_w);
  Espresso::Layer output_layer =
    Espresso::Eltwise("max", Espresso::Eltwise("prod", Espresso::Eltwise("sum", input_layer, input_layer), input_layer), input_layer);

  Halide::Image<float> input_(input_x, input_y, input_z, input_w);
  Espresso::fill_random(input_, generator, 0.0f, 1.0f);

  input.set(input_);

  // LOG(INFO) << input_;

  return run_net(output_layer, true);
}

double test_concat(std::default_random_engine& generator,
    int input_x=5, int input_y=7, int input_z=2, int input_w=3) {
  Halide::ImageParam input(Halide::type_of<float>(), 4);

  Espresso::Layer input_layer = Espresso::MemoryData(input, input_x, input_y, input_z, input_w);
  Espresso::Layer xwise_layer = Espresso::Concat(0, {input_layer, input_layer, input_layer});
  Espresso::Layer ywise_layer = Espresso::Concat(1, {xwise_layer, xwise_layer});
  Espresso::Layer zwise_layer = Espresso::Concat(2, {ywise_layer, ywise_layer});
  Espresso::Layer output_layer = Espresso::Concat(3, {zwise_layer});

  Halide::Image<float> input_(input_x, input_y, input_z, input_w);
  Espresso::fill_random(input_, generator, 0.0f, 1.0f);

  input.set(input_);

  // LOG(INFO) << input_;

  return run_net(output_layer, true);
}

double test_ffnn(std::default_random_engine& generator,
    int n_input=5, int n_hidden=7, int n_classes=3) {
  // construct abstract network
  Halide::ImageParam input(Halide::type_of<float>(), 4);
  Halide::ImageParam W1(Halide::type_of<float>(), 2);
  Halide::ImageParam b1(Halide::type_of<float>(), 1);
  Halide::ImageParam W2(Halide::type_of<float>(), 2);
  Halide::ImageParam b2(Halide::type_of<float>(), 1);

  Espresso::Layer input_layer = Espresso::MemoryData(input, n_input, 1, 1, 1);
  Espresso::Layer layer1 = Espresso::ReLU(Espresso::InnerProduct(input_layer, Halide::Func(W1), Halide::Func(b1), n_hidden));
  Espresso::Layer layer2 = Espresso::Softmax(Espresso::InnerProduct(layer1, Halide::Func(W2), Halide::Func(b2), n_classes));

  // instantiate inputs
  Halide::Image<float> input_(n_input);
  Halide::Image<float> W1_(n_hidden, n_input), W2_(n_classes, n_hidden);
  Halide::Image<float> b1_(n_hidden), b2_(n_classes);
  Espresso::fill_random(input_, generator, 0.0f, 1.0f);
  Espresso::fill_random(W1_, generator, 0.0f, 1.0f);
  Espresso::fill_random(W2_, generator, 0.0f, 1.0f);
  Espresso::fill_random(b1_, generator, 0.0f, 0.0f);
  Espresso::fill_random(b2_, generator, 0.0f, 0.0f);

  input.set(input_);
  W1.set(W1_);
  W2.set(W2_);
  b1.set(b1_);
  b2.set(b2_);

  // LOG(INFO) << input_;
  // LOG(INFO) << W1_;
  // LOG(INFO) << W2_;

  return run_net(layer2, true);
}

int test_main(std::vector<std::string>& input_ims) {
  std::random_device rd;
  std::default_random_engine generator(rd());

  std::vector<std::string> classes;
  std::ifstream infile("./images/meta/synset_words.txt");

  std::string line;
  while (std::getline(infile, line)) {
    classes.push_back(line);
  }

  std::string mean_proto = "./images/meta/imagenet_mean.binaryproto";
  BlobProto mean_blob;

  ReadProtoFromBinaryFile(mean_proto, &mean_blob);

  int count = input_ims.size();
  Halide::Image<float> mean_im = from_blob(mean_blob);
  Halide::ImageParam input_data(Halide::type_of<float>(), 4);
  Halide::Image<float> input_data_(227, 227, 3, count);

  for (int l = 0; l < count; l++) {
    Halide::Image<float> im = load_ppm<float>(input_ims[l]);
    for (int k = 0; k < 3; k++) {
      for (int j = 0; j < 227; j++) {
        for (int i = 0; i < 227; i++) {
          input_data_(i, j, k, l) = 256.0f * im(i + 15, j + 15, k) - mean_im(i + 15, j + 15, k);
        }
      }
    }
  }

  Halide::ImageParam input_labels(Halide::type_of<int>(), 4);
  input_data.set(input_data_);
  input_labels.set(Espresso::fill_random(Halide::Image<int>(1, 1, 1, count), generator, 500, 166)); // too lazy to initialize properly
  Espresso::Layer net = bvlc_reference_caffenet(input_data, input_labels, count);

  Halide::Image<float> output = run_net_with_output(net, true, count);

  for (int l = 0; l < count; l++) {
    std::vector<std::tuple<float, int> > results;
    for (int i = 0; i < 1000; i++) {
      results.push_back(std::make_pair(output(i, 0, 0, l), i));
    }
    std::sort(results.begin(), results.end());

    std::cout << "\n=== " << input_ims[l] << " ===\n";

    for (size_t i = 0; i < 10; i++) {
      float pct;
      int cls;
      std::tie(pct, cls) = results[1000-1-i];
      std::cout << "[" << i << "] " << classes[cls] << " (" << 100 * pct << "%)\n";
    }
  }

  return 0;
}

} // namespace Espresso
