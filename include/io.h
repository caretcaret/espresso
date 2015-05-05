#ifndef ESPRESSO_IO_H
#define ESPRESSO_IO_H

#include "common.h"

namespace Espresso {

using google::protobuf::Message;
using std::string;

bool ReadProtoFromTextFile(const string& filename, Message* proto);
void WriteProtoToTextFile(const Message& proto, const string& filename);

bool ReadProtoFromBinaryFile(const string& filename, Message* proto);
void WriteProtoToBinaryFile(const Message& proto, const string& filename);

void ReadNetParamsFromTextFile(const string& param_file, NetParameter* param);
void ReadSolverParamsFromTextFile(const string& param_file, SolverParameter* param);

} //namespace Espresso

#endif // ESPRESSO_IO_H
