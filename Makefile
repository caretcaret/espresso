.PHONY: clean build

all: build
build:
	clang++ -O2 -Wall -Werror -std=c++11 -Iinclude -Lbin -lHalide src/espresso.cpp -o build/espresso
run: build
	build/espresso
clean:
	rm build/*
