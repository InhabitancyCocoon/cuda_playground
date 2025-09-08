/*

bf16 matmul

warp level mma
warp group level mma

M = N = K = 512

compile command:


ref link:

https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions

https://github.com/tgautam03/tGeMM/blob/master/src/naive_tensor_tgemm.cu

https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__BFLOAT16.html#

https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/

*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <mma.h>
#include <cuda_bf16.h>

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

__global__ void mma_kernel() {

}


int main(int argc, char* argv[]) {
    
    return 0;
}