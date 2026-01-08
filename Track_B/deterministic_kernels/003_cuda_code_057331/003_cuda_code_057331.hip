#include <iostream>
#include <hip/hip_runtime.h>

// Kernel to perform parallel reduction to find the sum of elements
__global__ void sumKernel(float* d_input, float* d_output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load shared mem from global mem
    sdata[tid] = (i < n) ? d_input[i] : 0.0f;
    __syncthreads();

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) d_output[blockIdx.x] = sdata[0];
}

// Kernel to compute the mean of elements
__global__ void meanKernel(float* d_input, float* d_mean, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load shared mem from global mem
    sdata[tid] = (i < n) ? d_input[i] : 0.0f;
    __syncthreads();

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) d_mean[blockIdx.x] = sdata[0];
}

// Kernel to compute the variance
__global__ void varianceKernel(float* d_input, float mean, float* d_variance, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load shared mem from global mem
    sdata[tid] = (i < n) ? (d_input[i] - mean) * (d_input[i] - mean) : 0.0f;
    __syncthreads();

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) d_variance[blockIdx.x] = sdata[0];
}

int main() {
    const int N = 1024;
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    float* h_input = new float[N];
    float* d_input;
    float* d_intermediate;
    float* d_mean;
    float* d_variance;
    float h_mean, h_variance;

    // Initialize input array on host
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    hipMalloc((void**)&d_input, N * sizeof(float));
    hipMalloc((void**)&d_intermediate, numBlocks * sizeof(float));
    hipMalloc((void**)&d_mean, sizeof(float));
    hipMalloc((void**)&d_variance, numBlocks * sizeof(float));

    // Asynchronous memory copy from host to device
    hipMemcpyAsync(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice, 0);

    // Launch kernel to compute the sum
    sumKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_intermediate, N);
    hipDeviceSynchronize(); // Ensure sum is computed before mean

    // Launch kernel to compute the mean
    meanKernel<<<1, blockSize, blockSize * sizeof(float)>>>(d_intermediate, d_mean, numBlocks);
    hipMemcpyAsync(&h_mean, d_mean, sizeof(float), hipMemcpyDeviceToHost, 0);

    // Launch kernel to compute the variance
    varianceKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, h_mean, d_intermediate, N);
    hipDeviceSynchronize(); // Ensure variance computation is complete

    // Reduce the variance results
    meanKernel<<<1, blockSize, blockSize * sizeof(float)>>>(d_intermediate, d_variance, numBlocks);
    hipMemcpyAsync(&h_variance, d_variance, sizeof(float), hipMemcpyDeviceToHost, 0);

    // Wait for all asynchronous operations to complete
    hipDeviceSynchronize();

    // Free device memory
    hipFree(d_input);
    hipFree(d_intermediate);
    hipFree(d_mean);
    hipFree(d_variance);

    // Free host memory
    delete[] h_input;

    // Output results
    std::cout << "Mean: " << h_mean << std::endl;
    std::cout << "Variance: " << h_variance / N << std::endl;

    return 0;
}