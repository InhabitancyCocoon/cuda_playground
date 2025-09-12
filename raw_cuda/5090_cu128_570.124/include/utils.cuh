#pragma once

#include <iostream>
#include <assert.h>
#include <cuda_bf16.h>
#include <random>


// ref link: https://github.com/tgautam03/tGeMM/

// CUDA Error Checking
#define cuda_check(err) { \
    if (err != cudaSuccess) { \
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << "\n"; \
        exit(EXIT_FAILURE); \
    } \
}


// CUBLAS Error Checking
#define cublas_check(status) { \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Initializing matrix with given value
void init_mat(float* mat, int M, int N, float value);
void init_mat(nv_bfloat16* mat, int M, int N, nv_bfloat16 value);
void naive_gt_matmul(nv_bfloat16* A, nv_bfloat16* B, float* C, int M, int N, int K);

void random_init_mat(float* mat, int M, int N, int MIN, int MAX);
void random_init_mat(nv_bfloat16* mat, int M, int N, int MIN, int MAX);

void assert_mat_close(float* actual, float* expected, int M, int N, float atol, float rtol);
