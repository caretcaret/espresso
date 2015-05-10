#ifndef ESPRESSO_IMAGE_UTIL_H
#define ESPRESSO_IMAGE_UTIL_H

#include "Halide.h"
#include <random>

namespace Espresso {

template<class T = float>
Halide::Image<T> from_blob(const BlobProto& blob) {
    const BlobShape& shape = blob.shape();
    int num = shape.dim(0);
    int channels = shape.dim(1);
    int height = shape.dim(2);
    int width = shape.dim(3);
    int num_stride = channels * height * width;
    int channels_stride = height * width;
    int height_stride = width;
    Halide::Image<T> arr = Halide::Image<T>(height, width, channels, num);

    for (int l = 0; l < num; l++) {
        for (int k = 0; k < channels; k++) {
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    arr(i, j, k, l) = blob.data(l * num_stride + k * channels_stride + j * height_stride + i);
                }
            }
        }
    }

    return arr;
}

template<class T>
Halide::Image<T> fill_random(Halide::Image<T> arr, std::default_random_engine& generator, T mean, T stddev) {
    int dim0 = arr.width();
    int dim1 = arr.height();
    int dim2 = arr.channels();
    int dim3 = 1;
    int dims = arr.dimensions();

    if (dims >= 4) {
        dim3 = arr.extent(3);
    }

    std::normal_distribution<double> dist(mean, stddev);

    for (int i3 = 0; i3 < dim3; i3++) {
        for (int i2 = 0; i2 < dim2; i2++) {
            for (int i1 = 0; i1 < dim1; i1++) {
                for (int i0 = 0; i0 < dim0; i0++) {
                    arr(i0, i1, i2, i3) = (T) dist(generator);
                }
            }
        }
    }

    return arr;
}

} // namespace

template<class T>
std::ostream& operator<<(std::ostream& stream, const Halide::Image<T>& arr) {
    int dim0 = arr.width();
    int dim1 = arr.height();
    int dim2 = arr.channels();
    int dim3 = 1;
    int dims = arr.dimensions();

    if (dims >= 4) {
        dim3 = arr.extent(3);
        stream << "[";
    }
    for (int i3 = 0; i3 < dim3; i3++) {
        if (dims >= 3) {
            stream << (i3 == 0 ? "[" : ",\n[");
        }
        for (int i2 = 0; i2 < dim2; i2++) {
            if (dims >= 2) {
                stream << (i2 == 0 ? "[" : ",\n[");
            }
            for (int i1 = 0; i1 < dim1; i1++) {
                stream << (i1 == 0 ? "[" : ",\n[");
                for (int i0 = 0; i0 < dim0; i0++) {
                    stream << (i0 == 0 ? "" : ", ") << arr(i0, i1, i2, i3);
                }
                stream << "]";
            }
            if (dims >= 2 && i2 == dim2 - 1) {
                stream << "]";
            }
        }
        if (dims >= 3 && i3 == dim3 - 1) {
            stream << "]";
        }
    }
    if (dims >= 4) {
        stream << "]";
    }

    return stream;
}

#endif // ESPRESSO_IMAGE_UTIL_H
