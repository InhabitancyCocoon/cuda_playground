#include "../include/utils.cuh"


void init_mat(float* mat, int M, int N, float value) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            mat[row * N + col] = value;
        }
    }
}


void init_mat(nv_bfloat16* mat, int M, int N, nv_bfloat16 value) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            mat[row * N + col] = value;
        }
    }
}


// A, B, C are row-major. A (M x K) @ B (K x N) = C (M x N)
// C is zero-initialized.
void naive_gt_matmul(nv_bfloat16* A, nv_bfloat16* B, float* C, int M, int N, int K) {
    // for better locality
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                // C[i][j] += A[i][k] * B[k][j]
                C[i * N + j] += float(A[i * K + k] * B[k * N + j]);
            }
        }
    }
}


void random_init_mat(float* mat, int M, int N, int MIN, int MAX) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            mat[row * N + col] = float(rand() % (MAX - MIN + 1) + MIN);
        }
    }
}


void random_init_mat(nv_bfloat16* mat, int M, int N, int MIN, int MAX) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            mat[row * N + col] = float(rand() % (MAX - MIN + 1) + MIN);
        }
    }
}

// https://docs.pytorch.org/docs/stable/testing.html
void assert_mat_close(float* actual, float* expected, int M, int N, float atol, float rtol) {
    int mismatch = 0;
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float abs_diff = abs(actual[row * N + col] - expected[row * N + col]);
            if (abs_diff >= atol + rtol * expected[row * N + col]) {
                ++mismatch;
            }
        }
    }
    if (mismatch) {
        std::cerr << "Assert failure: " << mismatch << " / " << M * N << " is mismatched.\n";
    }
}