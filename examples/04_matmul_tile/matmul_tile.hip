// Tile-based GEMM kernel for capture demo.
// A: MxK, B: KxN, C: MxN (row-major)

#include <hip/hip_runtime.h>

#define TILE_M 16
#define TILE_N 16
#define K_BLOCK 16

extern "C" __global__ void matmul_tile(const float *A,
                                       const float *B,
                                       float *C,
                                       int M, int N, int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_M + ty;
    const int col = blockIdx.x * TILE_N + tx;

    __shared__ float As[TILE_M][K_BLOCK];
    __shared__ float Bs[K_BLOCK][TILE_N];

    float acc = 0.0f;

    // K dimension advances by K_BLOCK each iteration.
    for (int k0 = 0; k0 < K; k0 += K_BLOCK) {
        const int a_col = k0 + tx;
        const int b_row = k0 + ty;

        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
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
        C[row * N + col] = acc;
    }
}
