#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define BLOCK_SIZE 16
#define MATRIX_SIZE 64

__global__ void initializeMatrices(float* V, float* W, float* H, int m, int n, int k) {
    hiprandState state;
    hiprand_init(clock64(), threadIdx.x + blockIdx.x * blockDim.x, 0, &state);

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        V[row * n + col] = hiprand_uniform(&state) * 10.0f;
    }
    if (row < m && col < k) {
        W[row * k + col] = hiprand_uniform(&state) * 10.0f;
    }
    if (row < k && col < n) {
        H[row * n + col] = hiprand_uniform(&state) * 10.0f;
    }
}

__device__ float atomicAddFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val + __int_as_float(assumed)));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void nmfKernel(float* V, float* W, float* H, int m, int n, int k, float beta, float lambda) {
    extern __shared__ float sharedH[];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float V_ij = V[row * n + col];
        float WH_ij = 0.0f;

        // Load H into shared memory
        for (int l = threadIdx.x; l < k; l += blockDim.x) {
            sharedH[l] = H[l * n + col];
        }
        __syncthreads();

        // Compute WH
        for (int l = 0; l < k; ++l) {
            WH_ij += W[row * k + l] * sharedH[l];
        }

        float E_ij = V_ij - WH_ij;
        float numerator = 0.0f, denominator = 0.0f;

        // Update W
        for (int l = 0; l < k; ++l) {
            numerator += E_ij * H[l * n + col];
            denominator += WH_ij * H[l * n + col];
        }
        if (denominator != 0.0f) {
            atomicAddFloat(&W[row * k + col], beta * numerator / denominator - lambda * W[row * k + col]);
        }

        // Update H
        for (int l = 0; l < k; ++l) {
            numerator = 0.0f;
            denominator = 0.0f;
            for (int i = 0; i < m; ++i) {
                float WH_il = 0.0f;
                for (int j = 0; j < k; ++j) {
                    WH_il += W[i * k + j] * H[j * n + col];
                }
                numerator += W[row * k + l] * (V[i * n + col] - WH_il);
                denominator += W[row * k + l] * WH_il;
            }
            if (denominator != 0.0f) {
                atomicAddFloat(&H[l * n + col], beta * numerator / denominator - lambda * H[l * n + col]);
            }
        }
    }
}

int main() {
    int m = MATRIX_SIZE, n = MATRIX_SIZE, k = 32;
    float beta = 0.01f, lambda = 0.0001f;
    int iterations = 100;

    float* d_V, *d_W, *d_H;
    size_t size_V = m * n * sizeof(float);
    size_t size_W = m * k * sizeof(float);
    size_t size_H = k * n * sizeof(float);

    hipError_t err;

    err = hipMalloc(&d_V, size_V);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate V" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_W, size_W);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate W" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_H, size_H);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate H" << std::endl;
        return -1;
    }

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeMatrices<<<numBlocks, threadsPerBlock>>>(d_V, d_W, d_H, m, n, k);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize matrices" << std::endl;
        return -1;
    }
    hipDeviceSynchronize();

    for (int iter = 0; iter < iterations; ++iter) {
        nmfKernel<<<numBlocks, threadsPerBlock, k * sizeof(float)>>>(d_V, d_W, d_H, m, n, k, beta, lambda);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "NMF kernel launch failed: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
        hipDeviceSynchronize();
    }

    hipFree(d_V);
    hipFree(d_W);
    hipFree(d_H);

    return 0;
}