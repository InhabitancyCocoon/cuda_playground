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

https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation

https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/

*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <mma.h>
#include <cuda_bf16.h>
#include "../include/utils.cuh"
#include <cublas_v2.h>

// check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=wmma#element-types-and-matrix-sizes for details.
// template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;

// void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);
// void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
// void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
// void fill_fragment(fragment<...> &a, const T& v);
// void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);


constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

constexpr int M = 512, N = 128, K = 256;

__global__ void _naive_tc_mma_kernel(
    nv_bfloat16* d_A_ptr,
    nv_bfloat16* d_B_ptr,
    float* d_C_ptr,
    int M,
    int N,
    int K
) {
    int warp_M = blockIdx.x, warp_N = blockIdx.y; // think in warps way, not thread way.

    // don't know why VsCode fails to deduce this...
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;


    nvcuda::wmma::fill_fragment(c_frag, 0.f);

    int A_num_rows = M, A_num_cols = K;
    int B_num_rows = K, B_num_cols = N;
    int C_num_rows = M, C_num_cols = N;

    int A_row_stride = A_num_cols, A_col_stride = 1;
    int B_row_stride = B_num_cols, B_col_stride = 1;
    int C_row_stride = C_num_cols, C_col_stride = 1;

    int A_row_idx = warp_M * WMMA_M, A_col_idx = 0;
    int B_row_idx = 0, B_col_idx = warp_N * WMMA_N;
    int C_row_idx = warp_M * WMMA_M, C_col_idx = warp_N * WMMA_N;

    // move A, B, C pointers to the initial position.
    d_A_ptr += A_row_idx * A_row_stride + A_col_idx * A_col_stride;
    d_B_ptr += B_row_idx * B_row_stride + B_col_idx * B_col_stride;
    d_C_ptr += C_row_idx * C_row_stride + C_col_idx * C_col_stride;


    // tiled matmul, only this time tc takes the responsibility.
    // check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=wmma#wmma-description for details.
    for (int step = 0; step < K / WMMA_K; ++step) {

        // Waits until all warp lanes have arrived at load_matrix_sync and then loads the matrix fragment a from memory. 
        // mptr must be a 256-bit aligned pointer pointing to the first element of the matrix in memory. 
        // ldm describes the stride in elements between consecutive rows (for row major layout) or columns (for column major layout) 
        // and must be a multiple of 8 for __half element type or multiple of 4 for float element type. (i.e., multiple of 16 bytes in both cases). 
        // If the fragment is an accumulator, the layout argument must be specified as either mem_row_major or mem_col_major. 
        // For matrix_a and matrix_b fragments, the layout is inferred from the fragment’s layout parameter. 
        // The values of mptr, ldm, layout and all template parameters for a must be the same for all threads in the warp. 
        // This function must be called by all threads in the warp, or the result is undefined.
        nvcuda::wmma::load_matrix_sync(a_frag, d_A_ptr, A_num_cols);
        nvcuda::wmma::load_matrix_sync(b_frag, d_B_ptr, B_num_cols);


        // perform mma
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);


        // move A, B to the next wmma fragment
        d_A_ptr += WMMA_K * A_col_stride;
        d_B_ptr += WMMA_K * B_row_stride;
    }

    // store result into gmem
    nvcuda::wmma::store_matrix_sync(d_C_ptr, c_frag, C_num_cols, nvcuda::wmma::mem_row_major);

}

// tc works at warp level, different from cuda core which works at thread level logically.
void naive_tc_mma(
    nv_bfloat16* device_A,
    nv_bfloat16* device_B,
    float* device_C,
    int M,
    int N,
    int K
) {
    // performs nvcuda wmma, check 
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=wmma#warp-matrix-functions for details.

    dim3 block_size(32);
    dim3 grid_size(M / WMMA_M, N / WMMA_N);
    _naive_tc_mma_kernel<<<grid_size, block_size>>>(device_A, device_B, device_C, M, N, K);
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

    random_init_mat(host_A, M, K);
    random_init_mat(host_B, K, N);

    // init_mat(host_A, M, K, 1.f);
    // init_mat(host_B, K, N, 2.f);

    // init_mat_range(host_A, M, K);
    // init_mat_range(host_B, K, N);

    // print_mat(host_A, M, K);
    // print_mat(host_B, K, N);

    init_mat(host_C, M, N, 0);
    init_mat(gt_C, M, N, 0);

    naive_gt_matmul(host_A, host_B, gt_C, M, N, K);\
    print_mat(gt_C, M, N, "gt_C");

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
    The logic is now correct, but bf16 matmul behaves poorly than I expected in terms of precision.
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


    print_mat(host_C, M, N, "host_C cublas");
    // cublas now works.
    assert_mat_close(host_C, gt_C, M, N, 1e-3, 1e-1, "cublas");

    
    fill_device_matrix(device_C, M, N, 172.f);  // make sure the pre-computed result is erased.

    naive_tc_mma(device_A, device_B, device_C, M, N, K);

    cudaMemcpy(host_C, device_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    print_mat(host_C, M, N, "host_C naive tc mma");

    assert_mat_close(host_C, gt_C, M, N, 1e-3, 1e-1, "naive tc mma");

    return 0;
}