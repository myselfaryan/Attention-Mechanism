CXX = g++
CXXFLAGS = -std=c++23 -O2 -Wall -Wextra -Wconversion

all: attention_sequential attention_openmp attention_cuda attention_cuda_fused

attention_sequential: attention_sequential.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

attention_openmp: attention_openmp.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $^

attention_cuda: attention_cuda.cu
	$(CUDA_HOME)/bin/nvcc -O3 -o $@ $^

attention_cuda_fused: attention_cuda_fused.cu
	$(CUDA_HOME)/bin/nvcc -O3 -o $@ $^

clean:
	$(RM) attention_sequential attention_openmp attention_cuda attention_cuda_fused

.PHONY: all clean
