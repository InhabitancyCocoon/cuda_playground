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

void init_mat_range(float* mat, int M, int N) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            mat[row * N + col] = float(row * N + col);
        }
    }
}


void init_mat_range(nv_bfloat16* mat, int M, int N) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            mat[row * N + col] = float(row * N + col);
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


// fill the given matrix with random value from [0, 1)
void random_init_mat(float* mat, int M, int N) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float rand_float = (float)rand() / (float)RAND_MAX;
            mat[row * N + col] = rand_float;
        }
    }
}


// fill the given matrix with random value from [0, 1)
void random_init_mat(nv_bfloat16* mat, int M, int N) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float rand_float = (float)rand() / (float)RAND_MAX;
            mat[row * N + col] = (nv_bfloat16)rand_float;
        }
    }
}


// https://docs.pytorch.org/docs/stable/testing.html
void assert_mat_close(float* actual, float* expected, int M, int N, float atol, float rtol) {
    int total_elements = M * N;
    int mismatch = 0;

    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            int index = row * N + col;
            float a = actual[index];
            float e = expected[index];

            if (std::abs(a - e) >= atol + rtol * std::abs(e)) {
                ++mismatch;
            }
        }
    }

    if (mismatch) {
        double percentage = static_cast<double>(mismatch) / total_elements * 100.0;

        std::cerr << "Assert failure in file: " << __FILE__ 
                  << " line: " << __LINE__ << "\n"
                  << "Mismatched elements: " << mismatch << " / " << total_elements 
                  << " (" << std::fixed << std::setprecision(1) << percentage << "%)\n";
    } else {
        std::cout << "Congratulations! The assertion passed!\n";
    }
}



void print_mat(float* mat, int M, int N) {
    if (M > 16 || N > 16) {
        std::cout << "print_mat: M and N should be no greater than 16\n";
        return;
    }
    std::cout << M << " x " << N << " matrix: \n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << mat[i * N + j] << " ";
        }
        std::cout << "\n";
    }
}