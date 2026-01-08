#include <iostream>
#include <hip/hip_runtime.h>

#include <hip/hip_fp16.h>

#define BLOCK_SIZE 16

// Kernel for updating W matrix
__global__ void updateW(float* W, float* H, float* V, int m, int n, int k) {
    __shared__ float H_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float V_shared[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float W_value = 0.0f;

    // Loop over the W and H blocks
    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Load H and V into shared memory
        if (row < m && tile * BLOCK_SIZE + threadIdx.x < n) {
            H_shared[threadIdx.y][threadIdx.x] = H[(tile * BLOCK_SIZE + threadIdx.x) * k + col];
        } else {
            H_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (row < m && tile * BLOCK_SIZE + threadIdx.y < n) {
            V_shared[threadIdx.y][threadIdx.x] = V[row * n + (tile * BLOCK_SIZE + threadIdx.x)];
        } else {
            V_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Perform the computation
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            W_value += V_shared[threadIdx.y][i] * H_shared[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Update W matrix
    if (row < m && col < k) {
        float denominator = 0.0f;
        for (int i = 0; i < n; ++i) {
            denominator += H[i * k + col] * H[i * k + col];
        }
        W[row * k + col] *= W_value / denominator;
    }
}

// Kernel for updating H matrix
__global__ void updateH(float* W, float* H, float* V, int m, int n, int k) {
    __shared__ float W_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float V_shared[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float H_value = 0.0f;

    // Loop over the W and H blocks
    for (int tile = 0; tile < (m + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Load W and V into shared memory
        if (tile * BLOCK_SIZE + threadIdx.y < m && col < k) {
            W_shared[threadIdx.y][threadIdx.x] = W[(tile * BLOCK_SIZE + threadIdx.y) * k + col];
        } else {
            W_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (tile * BLOCK_SIZE + threadIdx.x < m && row < n) {
            V_shared[threadIdx.y][threadIdx.x] = V[(tile * BLOCK_SIZE + threadIdx.x) * n + row];
        } else {
            V_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Perform the computation
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            H_value += W_shared[i][threadIdx.x] * V_shared[threadIdx.y][i];
        }

        __syncthreads();
    }

    // Update H matrix
    if (row < n && col < k) {
        float denominator = 0.0f;
        for (int i = 0; i < m; ++i) {
            denominator += W[i * k + col] * W[i * k + col];
        }
        H[row * k + col] *= H_value / denominator;
    }
}

int main() {
    const int m = 128, n = 128, k = 64;
    const int num_iterations = 10;

    float* h_V = new float[m * n];
    float* h_W = new float[m * k];
    float* h_H = new float[n * k];

    // Initialize V, W, and H with random values
    for (int i = 0; i < m * n; ++i) h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < m * k; ++i) h_W[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < n * k; ++i) h_H[i] = static_cast<float>(rand()) / RAND_MAX;

    float* d_V, *d_W, *d_H;
    hipMalloc((void**)&d_V, m * n * sizeof(float));
    hipMalloc((void**)&d_W, m * k * sizeof(float));
    hipMalloc((void**)&d_H, n * k * sizeof(float));

    hipMemcpy(d_V, h_V, m * n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_W, h_W, m * k * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_H, h_H, n * k * sizeof(float), hipMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int iter = 0; iter < num_iterations; ++iter) {
        updateW<<<blocksPerGrid, threadsPerBlock>>>(d_W, d_H, d_V, m, n, k);
        hipDeviceSynchronize();
        updateH<<<blocksPerGrid, threadsPerBlock>>>(d_W, d_H, d_V, m, n, k);
        hipDeviceSynchronize();
    }

    hipMemcpy(h_W, d_W, m * k * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_H, d_H, n * k * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_V);
    hipFree(d_W);
    hipFree(d_H);

    // Free host memory
    delete[] h_V;
    delete[] h_W;
    delete[] h_H;

    return 0;
}