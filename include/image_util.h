#ifndef ESPRESSO_IMAGE_UTIL_H
#define ESPRESSO_IMAGE_UTIL_H

#include "Halide.h"
#include <random>
#include <glog/logging.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <algorithm>


#define _assert(condition, ...) if (!(condition)) {fprintf(stderr, __VA_ARGS__); exit(-1);}
#define SWAP_ENDIAN16(little_endian, value) if (little_endian) { (value) = (((value) & 0xff)<<8)|(((value) & 0xff00)>>8); }


namespace Espresso {

inline int is_little_endian() {
    int value = 1;
    return ((char *) &value)[0] == 1;
}
// Convert to u8
inline void convert(uint8_t in, uint8_t &out) {out = in;}
inline void convert(uint16_t in, uint8_t &out) {out = in >> 8;}
inline void convert(uint32_t in, uint8_t &out) {out = in >> 24;}
inline void convert(int8_t in, uint8_t &out) {out = in;}
inline void convert(int16_t in, uint8_t &out) {out = in >> 8;}
inline void convert(int32_t in, uint8_t &out) {out = in >> 24;}
inline void convert(float in, uint8_t &out) {out = (uint8_t)(in*255.0f);}
inline void convert(double in, uint8_t &out) {out = (uint8_t)(in*255.0f);}

// Convert to u16
inline void convert(uint8_t in, uint16_t &out) {out = in << 8;}
inline void convert(uint16_t in, uint16_t &out) {out = in;}
inline void convert(uint32_t in, uint16_t &out) {out = in >> 16;}
inline void convert(int8_t in, uint16_t &out) {out = in << 8;}
inline void convert(int16_t in, uint16_t &out) {out = in;}
inline void convert(int32_t in, uint16_t &out) {out = in >> 16;}
inline void convert(float in, uint16_t &out) {out = (uint16_t)(in*65535.0f);}
inline void convert(double in, uint16_t &out) {out = (uint16_t)(in*65535.0f);}

// Convert from u8
inline void convert(uint8_t in, uint32_t &out) {out = in << 24;}
inline void convert(uint8_t in, int8_t &out) {out = in;}
inline void convert(uint8_t in, int16_t &out) {out = in << 8;}
inline void convert(uint8_t in, int32_t &out) {out = in << 24;}
inline void convert(uint8_t in, float &out) {out = in/255.0f;}
inline void convert(uint8_t in, double &out) {out = in/255.0f;}

// Convert from u16
inline void convert(uint16_t in, uint32_t &out) {out = in << 16;}
inline void convert(uint16_t in, int8_t &out) {out = in >> 8;}
inline void convert(uint16_t in, int16_t &out) {out = in;}
inline void convert(uint16_t in, int32_t &out) {out = in << 16;}
inline void convert(uint16_t in, float &out) {out = in/65535.0f;}
inline void convert(uint16_t in, double &out) {out = in/65535.0f;}

template<typename T>
Halide::Image<T> load_ppm(std::string filename) {

    /* open file and test for it being a ppm */
    FILE *f = fopen(filename.c_str(), "rb");
    _assert(f, "File %s could not be opened for reading\n", filename.c_str());

    int width, height, maxval;
    char header[256];
    _assert(fscanf(f, "%255s", header) == 1, "Could not read PPM header\n");
    _assert(fscanf(f, "%d %d\n", &width, &height) == 2, "Could not read PPM width and height\n");
    _assert(fscanf(f, "%d", &maxval) == 1, "Could not read PPM max value\n");
    _assert(fgetc(f) != EOF, "Could not read char from PPM\n");

    int bit_depth = 0;
    if (maxval == 255) { bit_depth = 8; }
    else if (maxval == 65535) { bit_depth = 16; }
    else { _assert(false, "Invalid bit depth in PPM\n"); }

    _assert(strcmp(header, "P6") == 0 || strcmp(header, "p6") == 0, "Input is not binary PPM\n");

    int channels = 3;
    Halide::Image<T> im(width, height, channels);

    // convert the data to T
    if (bit_depth == 8) {
        uint8_t *data = new uint8_t[width*height*3];
        _assert(fread((void *) data,
                      sizeof(uint8_t), width*height*3, f) == (size_t) (width*height*3),
                "Could not read PPM 8-bit data\n");
        fclose(f);

        T *im_data = (T*) im.data();
        for (int y = 0; y < im.height(); y++) {
            uint8_t *row = (uint8_t *)(&data[(y*width)*3]);
            for (int x = 0; x < im.width(); x++) {
                convert(*row++, im_data[(0*height+y)*width+x]);
                convert(*row++, im_data[(1*height+y)*width+x]);
                convert(*row++, im_data[(2*height+y)*width+x]);
            }
        }
        delete[] data;
    } else if (bit_depth == 16) {
        int little_endian = is_little_endian();
        uint16_t *data = new uint16_t[width*height*3];
        _assert(fread((void *) data, sizeof(uint16_t), width*height*3, f) == (size_t) (width*height*3), "Could not read PPM 16-bit data\n");
        fclose(f);
        T *im_data = (T*) im.data();
        for (int y = 0; y < im.height(); y++) {
            uint16_t *row = (uint16_t *) (&data[(y*width)*3]);
            for (int x = 0; x < im.width(); x++) {
                uint16_t value;
                value = *row++; SWAP_ENDIAN16(little_endian, value); convert(value, im_data[(0*height+y)*width+x]);
                value = *row++; SWAP_ENDIAN16(little_endian, value); convert(value, im_data[(1*height+y)*width+x]);
                value = *row++; SWAP_ENDIAN16(little_endian, value); convert(value, im_data[(2*height+y)*width+x]);
            }
        }
        delete[] data;
    }
    im(0,0,0) = im(0,0,0);      /* Mark dirty inside read/write functions. */

    return im;
}

template<class T = float>
Halide::Image<T> transpose(Halide::Image<T> arr) {
    Halide::Image<T> arr2(arr.height(), arr.width());

    for (int j = 0; j < arr.height(); j++) {
      for (int i = 0; i < arr.width(); i++) {
        arr2(j, i) = arr(i, j);
      }
    }

    return arr2;
}

template<class T = float>
Halide::Image<T> from_blob(const BlobProto& blob, int dim=0) {
    int num = blob.has_num() ? blob.num() : 1;
    int channels = blob.has_channels() ? blob.channels() : 1;
    int height = blob.has_height() ? blob.height() : 1;
    int width = blob.has_width() ? blob.width() : 1;
    int num_stride = channels * height * width;
    int channels_stride = height * width;
    int height_stride = width;


    Halide::Image<T> arr;

    if (dim == 0) {
      if (width == 1) dim = 1;
      else if (height == 1) dim = 2;
      else if (channels == 1) dim = 3;
      else if (num == 1) dim = 4;
      else dim = 4;
    }
    
    if (dim == 1) arr = Halide::Image<T>(width);
    else if (dim == 2) arr = Halide::Image<T>(width, height);
    else if (dim == 3) arr = Halide::Image<T>(width, height, channels);
    else arr = Halide::Image<T>(width, height, channels, num);  // dim = 4

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
