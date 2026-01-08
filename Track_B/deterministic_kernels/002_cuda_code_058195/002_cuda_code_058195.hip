#include <iostream>
#include <hip/hip_runtime.h>
#include <math.h>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// CUDA kernel to compute the dot product of two vectors using shared memory
__global__ void dotProductKernel(const float* a, const float* b, float* result, int n) {
    extern __shared__ float sharedMem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float dot = 0.0f;

    // Load data into shared memory
    if (i < n) {
        sharedMem[tid] = a[i] * b[i];
    } else {
        sharedMem[tid] = 0.0f;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedMem[tid] += sharedMem[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, sharedMem[0]);
    }
}

// CUDA kernel to compute the sum of squares of a vector using shared memory
__global__ void sumSquaresKernel(const float* a, float* result, int n) {
    extern __shared__ float sharedMem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // Load data into shared memory
    if (i < n) {
        sharedMem[tid] = a[i] * a[i];
    } else {
        sharedMem[tid] = 0.0f;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedMem[tid] += sharedMem[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, sharedMem[0]);
    }
}

int main() {
    const int n = 64;
    const int blockSize = 32;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    // Initialize host vectors
    thrust::host_vector<float> h_a(n);
    thrust::host_vector<float> h_b(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(n - i);
    }

    // Allocate device memory
    float* d_a, *d_b, *d_dot, *d_sumSquaresA, *d_sumSquaresB;
    hipMalloc(&d_a, n * sizeof(float));
    hipMalloc(&d_b, n * sizeof(float));
    hipMalloc(&d_dot, sizeof(float));
    hipMalloc(&d_sumSquaresA, sizeof(float));
    hipMalloc(&d_sumSquaresB, sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_a, h_a.data(), n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b.data(), n * sizeof(float), hipMemcpyHostToDevice);

    // Initialize reduction results to 0
    hipMemset(d_dot, 0, sizeof(float));
    hipMemset(d_sumSquaresA, 0, sizeof(float));
    hipMemset(d_sumSquaresB, 0, sizeof(float));

    // Launch dot product kernel
    dotProductKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_a, d_b, d_dot, n);
    hipDeviceSynchronize();

    // Launch sum squares kernels
    sumSquaresKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_a, d_sumSquaresA, n);
    sumSquaresKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_b, d_sumSquaresB, n);
    hipDeviceSynchronize();

    // Copy results back to host
    float dot, sumSquaresA, sumSquaresB;
    hipMemcpy(&dot, d_dot, sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(&sumSquaresA, d_sumSquaresA, sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(&sumSquaresB, d_sumSquaresB, sizeof(float), hipMemcpyDeviceToHost);

    // Calculate cosine similarity
    float cosineSimilarity = dot / (sqrtf(sumSquaresA) * sqrtf(sumSquaresB));

    // Print results
    std::cout << "Dot Product: " << dot << std::endl;
    std::cout << "Sum of Squares A: " << sumSquaresA << std::endl;
    std::cout << "Sum of Squares B: " << sumSquaresB << std::endl;
    std::cout << "Cosine Similarity: " << cosineSimilarity << std::endl;

    // Free device memory
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_dot);
    hipFree(d_sumSquaresA);
    hipFree(d_sumSquaresB);

    return 0;
}