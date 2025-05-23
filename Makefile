CXX = g++-11
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3
NVCCFLAGS = -O3 -gencode arch=compute_86,code=sm_86 -I.
LDFLAGS = -lpthread -lcurl -lgmp

OBJ = main.o stratum_client.o dag_generator.o blake2b.o blake2b_cuda.o autolykos2_cuda_miner.o

all: miner

main.o: main.cpp stratum_client.h autolykos2_cuda_miner.cuh
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

stratum_client.o: stratum_client.cpp stratum_client.h job.h utils.h
	$(CXX) $(CXXFLAGS) -c stratum_client.cpp -o stratum_client.o

dag_generator.o: dag_generator.cpp dag_generator.h
	$(CXX) $(CXXFLAGS) -c dag_generator.cpp -o dag_generator.o

blake2b.o: blake2b.c blake2b.h
	cc -c -o blake2b.o blake2b.c

blake2b_cuda.o: blake2b_cuda.cu blake2b_cuda.cuh
	$(NVCC) -ccbin $(CXX) $(NVCCFLAGS) -c blake2b_cuda.cu -o blake2b_cuda.o

autolykos2_cuda_miner.o: autolykos2_cuda_miner.cu autolykos2_cuda_miner.cuh blake2b_cuda.cuh
	$(NVCC) -ccbin $(CXX) $(NVCCFLAGS) -c autolykos2_cuda_miner.cu -o autolykos2_cuda_miner.o

miner: $(OBJ)
	$(NVCC) -ccbin $(CXX) $(NVCCFLAGS) -o miner $(OBJ) $(LDFLAGS)

clean:
	rm -f *.o miner
