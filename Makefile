.PHONY: clean build

all: build
build:
	g++ -std=c++11 -Iinclude -Lbin -lHalide src/espresso.cpp -o build/espresso
run: build
	build/espresso
clean:
	rm build/*
