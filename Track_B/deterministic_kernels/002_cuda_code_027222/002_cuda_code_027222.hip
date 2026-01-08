#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void parallelReduceAverage(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    // Each block processes one segment of the array
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load shared mem from global mem
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void computeAverage(float* partialSums, float* average, int numBlocks) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;

    // Load shared mem from global mem
    sdata[tid] = (tid < numBlocks) ? partialSums[tid] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) {
        *average = sdata[0] / numBlocks;
    }
}

int main() {
    const int N = 4096;
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    // Allocate host memory
    float* h_input = new float[N];
    float h_average;

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_input;
    float* d_partialSums;
    float* d_average;
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_partialSums, numBlocks * sizeof(float));
    hipMalloc(&d_average, sizeof(float));

    // Copy input data from host to device
    hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice);

    // Launch parallel reduction kernel
    parallelReduceAverage<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_partialSums, N);
    hipDeviceSynchronize();

    // Check for errors in kernel launch
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch kernel to compute average from partial sums
    computeAverage<<<1, blockSize, blockSize * sizeof(float)>>>(d_partialSums, d_average, numBlocks);
    hipDeviceSynchronize();

    // Check for errors in kernel launch
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy result back to host
    hipMemcpy(&h_average, d_average, sizeof(float), hipMemcpyDeviceToHost);

    // Print the result
    std::cout << "Computed Average: " << h_average << std::endl;

    // Free device memory
    hipFree(d_input);
    hipFree(d_partialSums);
    hipFree(d_average);

    // Free host memory
    delete[] h_input;

    return 0;
}