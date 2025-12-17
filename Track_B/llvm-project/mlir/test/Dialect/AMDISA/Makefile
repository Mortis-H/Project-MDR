HIPCC ?= /opt/rocm/bin/hipcc
CXXFLAGS = -std=c++11 -O2

all: hsaco_runner universal_hsaco_runner

hsaco_runner: hsaco_runner.cpp
	$(HIPCC) $(CXXFLAGS) $< -o $@

universal_hsaco_runner: universal_hsaco_runner.cpp
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f hsaco_runner universal_hsaco_runner

.PHONY: all clean

