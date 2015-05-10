CC = clang++

PROJDIRS := src include build/gen

SRCFILES := $(shell find $(PROJDIRS) -type f -name '*.cc' 2>/dev/null)
HDRFILES := $(shell find $(PROJDIRS) -type f -name '*.h' 2>/dev/null)

OBJFILES := $(patsubst %.cc,%.o,$(SRCFILES))

DEPFILES := $(patsubst %.cc,%.d,$(SRCFILES))

ALLFILES := $(SRCFILES) $(HDRFILES)

PROTO_SRC = $(shell find proto -type f -name '*.proto')

LIBS= -lHalide -lprotobuf -lglog
LIBDIRS := -Lbin -L/usr/local/lib -L/usr/local/bin

override CXXFLAGS += -O2 -g -Wall -Wextra -std=c++11 $(addprefix -I, $(PROJDIRS))
override LDFLAGS += $(LIBDIRS) $(LIBS)


.PHONY: clean 

all: prebuild proto
	@$(MAKE) -s espresso
prebuild:
	@mkdir -p build build/gen
espresso: $(OBJFILES)
	@$(CC) $(CXXFLAGS) $(LDFLAGS) $? -o build/espresso
proto: prebuild
	@protoc --cpp_out=./build/gen $(PROTO_SRC)
%.o: %.cc Makefile
	@$(CC) $(CXXFLAGS) -MMD -MP -c $< -o $@
run: all
	@build/espresso
debug: all
	@GLOG_logtostderr=1 build/espresso
clean:
	@-rm -rf src/*.d src/*.o build

-include $(DEPFILES)

