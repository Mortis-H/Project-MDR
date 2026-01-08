#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Kernel to perform parallel reduction using shared memory and atomic operations
__global__ void reduceKernel(const float* input, float* partialSums, int n) {
    extern __shared__ float sharedData[];

    // Load input into shared memory
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sharedData[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        partialSums[blockIdx.x] = sharedData[0];
    }
}

// Kernel to compute the mean and variance
__global__ void computeMeanVariance(const float* input, float* mean, float* variance, int n) {
    extern __shared__ float sharedData[];

    // Load input into shared memory
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sharedData[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory for sum and sum of squares
    float sum = 0.0f, sumOfSquares = 0.0f;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
            sharedData[tid + blockDim.x / 2] += sharedData[tid + blockDim.x / 2 + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(mean, sharedData[0]);
        atomicAdd(variance, sharedData[blockDim.x / 2]);
    }
}

int main() {
    const int n = 8192;
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    // Allocate host memory
    float* h_input = new float[n];
    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX; // Random values between 0 and 1
    }

    // Allocate device memory
    float* d_input;
    float* d_partialSums;
    float* d_mean;
    float* d_variance;
    hipMalloc((void**)&d_input, n * sizeof(float));
    hipMalloc((void**)&d_partialSums, numBlocks * sizeof(float));
    hipMalloc((void**)&d_mean, sizeof(float));
    hipMalloc((void**)&d_variance, sizeof(float));

    // Copy input data to device
    hipMemcpy(d_input, h_input, n * sizeof(float), hipMemcpyHostToDevice);

    // Initialize mean and variance to 0
    hipMemset(d_mean, 0, sizeof(float));
    hipMemset(d_variance, 0, sizeof(float));

    // Launch kernel to compute sum and sum of squares
    computeMeanVariance<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_mean, d_variance, n);
    hipDeviceSynchronize();

    // Copy results back to host
    float h_mean, h_variance;
    hipMemcpy(&h_mean, d_mean, sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(&h_variance, d_variance, sizeof(float), hipMemcpyDeviceToHost);

    // Compute mean and variance
    h_mean /= n;
    h_variance = h_variance / n - h_mean * h_mean;

    // Compute standard deviation
    float h_stddev = sqrtf(h_variance);

    // Print results
    std::cout << "Mean: " << h_mean << std::endl;
    std::cout << "Standard Deviation: " << h_stddev << std::endl;

    // Free device memory
    hipFree(d_input);
    hipFree(d_partialSums);
    hipFree(d_mean);
    hipFree(d_variance);

    // Free host memory
    delete[] h_input;

    return 0;
}