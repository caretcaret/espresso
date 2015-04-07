.PHONY: clean all

all: build
build:
	g++ -Iinclude -Lbin -lHalide src/espresso.cpp -o build/espresso
run: build
	build/espresso
clean:
	rm build/*
