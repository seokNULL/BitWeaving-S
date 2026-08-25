CXX      ?= g++
CXXFLAGS ?= -mavx2 -O2 -g -fopenmp
TARGET    = bitweaving

.PHONY: all clean

all: $(TARGET)

$(TARGET): bitweaving.cpp SIMD_operations.h
	$(CXX) $(CXXFLAGS) bitweaving.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)
