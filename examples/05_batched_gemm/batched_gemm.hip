// Batched GEMM kernel with runtime stride/leading dimension.
// A: [batch, M, K], B: [batch, K, N], C: [batch, M, N] (row-major)

#include <hip/hip_runtime.h>
#include <stdint.h>

#define TILE_M 16
#define TILE_N 16
#define K_BLOCK 16

extern "C" __global__ void batched_gemm(const float *A,
                                        const float *B,
                                        float *C,
                                        int M, int N, int K,
                                        int lda, int ldb, int ldc,
                                        int64_t strideA, int64_t strideB, int64_t strideC,
                                        int batchCount) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_M + ty;
    const int col = blockIdx.x * TILE_N + tx;
    const int batch = blockIdx.z;

    if (batch >= batchCount) {
        return;
    }

    __shared__ float As[TILE_M][K_BLOCK];
    __shared__ float Bs[K_BLOCK][TILE_N];

    const float *A_base = A + static_cast<int64_t>(batch) * strideA;
    const float *B_base = B + static_cast<int64_t>(batch) * strideB;
    float *C_base = C + static_cast<int64_t>(batch) * strideC;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += K_BLOCK) {
        const int a_col = k0 + tx;
        const int b_row = k0 + ty;

        if (row < M && a_col < K) {
            As[ty][tx] = A_base[row * lda + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (b_row < K && col < N) {
            Bs[ty][tx] = B_base[b_row * ldb + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < K_BLOCK; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C_base[row * ldc + col] = acc;
    }
}
