/*

bf16 matmul

warp level mma
warp group level mma

M = 512
N = 128
K = 256

matrix multiplication:
C = A @ B

A (M, K)
B (K, N)
C (M, N)

A non-transposed, row-major
B non-transposed, row-major
C non-transposed, row-major

ref link:

https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions

https://github.com/tgautam03/tGeMM/blob/master/src/naive_tensor_tgemm.cu

https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__BFLOAT16.html#

https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/

https://docs.nvidia.com/cuda/cublas/#cublasgemmex

https://docs.nvidia.com/cuda/cublas/#data-layout

https://blog.hedup.cc/?p=665

*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <mma.h>
#include <cuda_bf16.h>
#include "../include/utils.cuh"
#include <cublas_v2.h>

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int M = 512, N = 128, K = 256;

__global__ void mma_kernel() {

}


int main(int argc, char* argv[]) {
    nv_bfloat16 *host_A, *host_B;
    float *host_C, *gt_C;
    nv_bfloat16 *device_A, *device_B;
    float *device_C;

    host_A = (nv_bfloat16*)malloc(sizeof(nv_bfloat16) * M * K);
    host_B = (nv_bfloat16*)malloc(sizeof(float) * K * N);
    host_C = (float*)malloc(sizeof(float) * M * N);
    gt_C = (float*)malloc(sizeof(float) * M * N);

    cuda_check(cudaMalloc((void**)&device_A, sizeof(nv_bfloat16) * M * K));
    cuda_check(cudaMalloc((void**)&device_B, sizeof(nv_bfloat16) * K * N));
    cuda_check(cudaMalloc((void**)&device_C, sizeof(float) * M * N));

    random_init_mat(host_A, M, N, 0, 100);
    random_init_mat(host_B, M, N, 0, 100);
    init_mat(host_C, M, N, 0);
    init_mat(gt_C, M, N, 0);

    naive_gt_matmul(host_A, host_B, gt_C, M, N, K);

    cuda_check(cudaMemcpy(device_A, host_A, sizeof(nv_bfloat16) * M * K, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(device_B, host_B, sizeof(nv_bfloat16) * K * N, cudaMemcpyHostToDevice));


    float alpha = 1, beta = 0;
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle));

    /*
    This is kind of weird.
    The cuBLAS doc said:  A, B and C are matrices stored in column-major format with dimensions:
    A (M, K)
    B (K, N)
    C (M, N)
    But in C++ world, the matrix is typically stored in row-major form.
    To make the result C directly usable(in other words, stored in row-major form)
    in C++, we have to do some tricks.

    The basic idea is: C = (B^T @ A^T)^T = A @ B

    check linear.py within this directory for details.

    In a nutshell, this becomes:

    B.T (N x k) @ A.T (K x M) = C.T (N x M)

    In column-major, the leading dimension is the number of rows.

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n)
    */

    cublas_check(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        device_B, CUDA_R_16BF, N,
        device_A, CUDA_R_16BF, K,
        &beta,
        device_C, CUDA_R_32F, N,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    cudaMemcpy(host_C, device_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    assert_mat_close(host_C, gt_C, M, N, 1e-5, 1e-4);


    return 0;
}