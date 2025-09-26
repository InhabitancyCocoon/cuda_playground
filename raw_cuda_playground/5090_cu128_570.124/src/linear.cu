/*
This is a linear layer y = x @ A.T + b. Both forward pass and backward pass are implemented. I only
implement for square matrix matmul such as m512n512k512. The computation is done in bf16 precision
to utilize tensor core.

bwd formula:

Given dy,
dx = dy @ A
dA = dy.T @ x

The compile command:


The target is to utilize tensor core and use the kernel in pytorch.
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>


#define CUDA_CHECK(call) \
{                         \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s: %d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

constexpr int M = 512, N = 512, K = 512;


__global__ void _linear_kernel(float* y, float* x, float* A) {

}

__global__ void _linear_bwd_kernel(float* dy, float* x, float* A, float* dx, float* dA) {

}


int main(int argc, char* argv[]) {
    return 0;
}