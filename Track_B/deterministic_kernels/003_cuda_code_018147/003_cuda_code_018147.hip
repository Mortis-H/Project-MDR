#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define BLOCK_SIZE 16
#define MATRIX_SIZE 512

__global__ void initializeMatrices(float* V, float* W, float* H, int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < M && idy < N) {
        int index = idx * N + idy;
        V[index] = 1.0f + (float)idx * (float)idy / (float)(M * N);
    }

    if (idx < M && idy < K) {
        int index = idx * K + idy;
        W[index] = 1.0f + (float)idx * (float)idy / (float)(M * K);
    }

    if (idx < K && idy < N) {
        int index = idx * N + idy;
        H[index] = 1.0f + (float)idx * (float)idy / (float)(K * N);
    }
}

__global__ void updateW(float* V, float* W, float* H, int M, int N, int K, float alpha) {
    __shared__ float sharedH[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedW[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float Cvalue = 0.0f;

    for (int e = 0; e < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++e) {
        if (row < M && e * BLOCK_SIZE + threadIdx.x < N) {
            sharedH[threadIdx.y][threadIdx.x] = H[(e * BLOCK_SIZE + threadIdx.x) * K + col];
        } else {
            sharedH[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < K && e * BLOCK_SIZE + threadIdx.y < N) {
            sharedW[threadIdx.y][threadIdx.x] = W[row * K + e * BLOCK_SIZE + threadIdx.y];
        } else {
            sharedW[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += sharedW[threadIdx.y][k] * sharedH[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        float numerator = 0.0f;
        float denominator = 0.0f;

        for (int n = 0; n < N; ++n) {
            numerator += V[row * N + n] * H[n * K + col];
            denominator += W[row * K + col] * H[n * K + col];
        }

        W[row * K + col] *= (numerator / denominator) * alpha;
    }
}

__global__ void updateH(float* V, float* W, float* H, int M, int N, int K, float alpha) {
    __shared__ float sharedW[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedH[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float Cvalue = 0.0f;

    for (int e = 0; e < (M + BLOCK_SIZE - 1) / BLOCK_SIZE; ++e) {
        if (row < K && e * BLOCK_SIZE + threadIdx.x < M) {
            sharedW[threadIdx.y][threadIdx.x] = W[(e * BLOCK_SIZE + threadIdx.x) * K + row];
        } else {
            sharedW[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && e * BLOCK_SIZE + threadIdx.y < M) {
            sharedH[threadIdx.y][threadIdx.x] = H[col * K + e * BLOCK_SIZE + threadIdx.y];
        } else {
            sharedH[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += sharedW[threadIdx.y][k] * sharedH[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < K && col < N) {
        float numerator = 0.0f;
        float denominator = 0.0f;

        for (int m = 0; m < M; ++m) {
            numerator += W[m * K + row] * V[m * N + col];
            denominator += W[m * K + row] * H[row * N + col];
        }

        H[row * N + col] *= (numerator / denominator) * alpha;
    }
}

int main() {
    const int M = MATRIX_SIZE;
    const int N = MATRIX_SIZE;
    const int K = 64;
    const int numIterations = 100;
    const float alpha = 0.01f;

    float* h_V, *h_W, *h_H;
    float* d_V, *d_W, *d_H;

    size_t sizeV = M * N * sizeof(float);
    size_t sizeWH = M * K * sizeof(float);
    size_t sizeH = K * N * sizeof(float);

    h_V = (float*)malloc(sizeV);
    h_W = (float*)malloc(sizeWH);
    h_H = (float*)malloc(sizeH);

    hipMalloc((void**)&d_V, sizeV);
    hipMalloc((void**)&d_W, sizeWH);
    hipMalloc((void**)&d_H, sizeH);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    initializeMatrices<<<gridSize, blockSize>>>(d_V, d_W, d_H, M, N, K);
    hipDeviceSynchronize();

    for (int iter = 0; iter < numIterations; ++iter) {
        updateW<<<gridSize, blockSize>>>(d_V, d_W, d_H, M, N, K, alpha);
        updateH<<<gridSize, blockSize>>>(d_V, d_W, d_H, M, N, K, alpha);
        hipDeviceSynchronize();
    }

    hipMemcpy(h_W, d_W, sizeWH, hipMemcpyDeviceToHost);
    hipMemcpy(h_H, d_H, sizeH, hipMemcpyDeviceToHost);

    hipFree(d_V);
    hipFree(d_W);
    hipFree(d_H);

    free(h_V);
    free(h_W);
    free(h_H);

    return 0;
}