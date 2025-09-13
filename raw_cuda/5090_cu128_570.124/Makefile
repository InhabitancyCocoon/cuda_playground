CC = nvcc
VERBOSE = --ptxas-options=-v
LINK_CUBLAS = -lcublas

# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#device-c-dc
DEVICE_COMPILE_FLAG = -dc

# sm_75 is used as the default value for --gpu-architecture,
# PTX is generated for compute_75 then assembled and optimized for sm_75.

.PHONY: clean

build/bf16_demo.out: src/bf16_demo.cu
	$(CC) src/bf16_demo.cu -o build/bf16_demo.out


# Utils
build/utils.o: src/utils.cu
	$(CC) $(DEVICE_COMPILE_FLAG) src/utils.cu -o build/utils.o


build/naive_tc_mma.out: src/naive_tc_mma.cu build/utils.o
	$(CC) ${LINK_CUBLAS} build/utils.o src/naive_tc_mma.cu -o build/naive_tc_mma.out



clean:
	@echo "Removing object files..."
	rm -f build/*.out
	rm -f build/*.o