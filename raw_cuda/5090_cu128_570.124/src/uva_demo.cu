/*
UVA (Unified Virtual Address) is a CUDA feature which puts all CUDA execution, host and GPUs, in the same address space.


ref link:

https://nichijou.co/cudaRandom-UVA/

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-virtual-address-space


*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <iostream>


__global__ void double_elements_kernel(float* dev_ptr, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        dev_ptr[idx] = -dev_ptr[idx];
    }
}


int main(int argc, char* argv[]) {

    float *host_ptr;

    cudaHostAlloc((void**)&host_ptr, sizeof(float) * 128, cudaHostAllocDefault);

    for (int i = 0; i < 128; ++i) {
        *(host_ptr + i) = i;
        std::cout << host_ptr[i] << " ";
    }

    std::cout << "\n";

    dim3 block_size(32);
    dim3 grid_size(128 / 32);

    double_elements_kernel<<<grid_size,block_size>>>(host_ptr, 128);


    for (int i = 0; i < 128; ++i) {
        std::cout << host_ptr[i] << " ";
    }

    std::cout << "\n";


    return 0;
}