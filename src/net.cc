#include "net.h"
#include "io.h"
#include "LayerFactory.h"

namespace Espresso {
Net::Net(const NetParameter& param) {
    init(param);
}

Net::Net(const string& filename) {
    NetParameter param;
    ReadNetParamsFromTextFile(filename, &param);
    init(param);
}

void Net::init(const NetParameter& param) {
    this->name = param.name();

    for (int i = 0; i < param.layer_size(); i++) {
        auto& layerParam = param.layer(i);
        this->layers.push_back(LayerFactory::create(layerParam));
    }
}


}
