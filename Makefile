# Makefile for Ergo GPU Miner with plain TCP (no SSL)
CXX = g++-11
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3
NVCCFLAGS = -ccbin /usr/bin/g++-11 -allow-unsupported-compiler -gencode arch=compute_86,code=sm_86 -O3
LDFLAGS = -lpthread

OBJS = main.o stratum_client.o dag_generator.o blake2b.o autolykos2_cuda_miner.o

all: miner

miner: $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f *.o miner
