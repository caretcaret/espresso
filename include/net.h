#ifndef ESPRESSO_NET_H
#define ESPRESSO_NET_H

#include "proto/caffe.pb.h"
#include "layer.h"
#include <vector>
#include <tuple>
#include <memory>
#include <string>

namespace Espresso {

using std::string;
using std::tuple;
using std::vector;
using std::unique_ptr;

class Net {
public:
    explicit Net(const NetParameter& param);
    explicit Net(const string& filename);

protected:
    string name;
    vector<unique_ptr<Layer>> layers;


private:
    void init(const NetParameter& param);
};

} // namespace Espresso

#endif // ESPRESSO_NET_H
