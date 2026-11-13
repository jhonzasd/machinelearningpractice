CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
ASAN_OPTIONS ?= detect_leaks=1
COMMON = src/regression.cpp src/dataset.cpp

.PHONY: all run test sanitize clean
all: build/regression_demo build/regression_tests

build:
	mkdir -p build

build/regression_demo: $(COMMON) src/main.cpp include/regression.hpp include/dataset.hpp | build
	$(CXX) $(CXXFLAGS) -Iinclude $(COMMON) src/main.cpp -o $@

build/regression_tests: $(COMMON) tests/test_regression.cpp include/regression.hpp include/dataset.hpp | build
	$(CXX) $(CXXFLAGS) -Iinclude $(COMMON) tests/test_regression.cpp -o $@

run: build/regression_demo
	./build/regression_demo data results

test: build/regression_tests
	./build/regression_tests data

sanitize: | build
	$(CXX) -std=c++17 -g -O1 -Wall -Wextra -Wpedantic -fsanitize=address,undefined -fno-omit-frame-pointer -Iinclude $(COMMON) tests/test_regression.cpp -o build/tests_sanitized
	ASAN_OPTIONS=$(ASAN_OPTIONS) ./build/tests_sanitized data

clean:
	rm -rf build
