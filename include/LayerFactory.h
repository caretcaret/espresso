#ifndef ESPRESSO_LAYERFACTORY_H
#define ESPRESSO_LAYERFACTORY_H

#include <functional>
#include <map>
#include <string>

#include <memory>

#include "common.h"
#include "layer.h"
#include "proto/caffe.pb.h"

// TODO: Add constructors to layer classes taking a LayerParameter as input and add
// REGISTER_LAYER_CLASS(layerclass) underneath the class definition

// I did this to Convolution, but the constructor doesn't do anything currently

namespace Espresso {

using std::string;
using std::function;
using std::unique_ptr;

using Creator = function<unique_ptr<Layer>(const LayerParameter&)>;
using CreatorRegistry = std::map<string, Creator>;

class LayerFactory {
public:
    static CreatorRegistry& Registry() {
        static CreatorRegistry registry;
        return registry;
    }

    static unique_ptr<Layer> create(const LayerParameter& param) {
        const string& type = param.type();
        auto& registry = Registry();
        CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
            << " (known types: " << LayerTypeList() << ")";

        return registry[type](param);
    }

    // Adds a creator.
    static void AddCreator(const string& type, Creator creator) {
        auto& registry = Registry();

        CHECK_EQ(registry.count(type), 0)
            << "Layer type " << type << " already registered.";
        registry[type] = creator;
    }

private:
    LayerFactory() {}

    static string LayerTypeList() {
        string layer_types;
        auto& registry = Registry();

        for (auto iter = registry.begin(); iter != registry.end(); ++iter) {
            if (iter != registry.begin()) {
                layer_types += ", ";
            }
            layer_types += iter->first;
        }
        return layer_types;
    }


};

class LayerRegisterer {
public:
    LayerRegisterer(const string& type, Creator creator) {
        LayerFactory::AddCreator(type, creator);
    }
};

#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer g_creator_##type(#type, creator);                    \

#define REGISTER_LAYER_CLASS(type)                                             \
  unique_ptr<Layer> Creator_##type(const LayerParameter& param)                    \
  {                                                                            \
    return unique_ptr<Layer>(new type(param));                                          \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type)



}   // namespace Espresso

#endif // ESPRESSO_LAYERFACTORY_H
