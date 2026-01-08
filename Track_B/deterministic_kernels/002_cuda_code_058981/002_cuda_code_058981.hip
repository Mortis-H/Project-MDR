#include <iostream>
#include <hip/hip_runtime.h>

// Define the size of the array
#define N 1024

// Kernel to compute the sum of an array using shared memory
__global__ void parallelSum(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    // Each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global mem
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Kernel to compute the sum of squares of an array using shared memory
__global__ void parallelSumSquares(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    // Each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] * input[i] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global mem
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main() {
    float *h_input, *d_input, *d_intermediate, *h_intermediate;
    float sum, sumSquares, variance;
    int numBlocks, blockSize;

    // Initialize host input data
    h_input = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i);
    }

    // Set block size and number of blocks
    blockSize = 256;
    numBlocks = (N + blockSize - 1) / blockSize;

    // Allocate device memory
    hipMalloc((void **)&d_input, N * sizeof(float));
    hipMalloc((void **)&d_intermediate, numBlocks * sizeof(float));

    // Copy input data from host to device
    hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice);

    // Allocate host memory for intermediate results
    h_intermediate = (float *)malloc(numBlocks * sizeof(float));

    // Launch kernel to compute the sum of the array
    parallelSum<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_intermediate, N);
    hipDeviceSynchronize();

    // Copy intermediate results from device to host
    hipMemcpy(h_intermediate, d_intermediate, numBlocks * sizeof(float), hipMemcpyDeviceToHost);

    // Perform final reduction on the host to get the total sum
    sum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        sum += h_intermediate[i];
    }

    // Launch kernel to compute the sum of squares of the array
    parallelSumSquares<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_intermediate, N);
    hipDeviceSynchronize();

    // Copy intermediate results from device to host
    hipMemcpy(h_intermediate, d_intermediate, numBlocks * sizeof(float), hipMemcpyDeviceToHost);

    // Perform final reduction on the host to get the total sum of squares
    sumSquares = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        sumSquares += h_intermediate[i];
    }

    // Compute variance
    variance = (sumSquares - (sum * sum) / N) / N;

    // Print the result
    std::cout << "Variance: " << variance << std::endl;

    // Free device memory
    hipFree(d_input);
    hipFree(d_intermediate);

    // Free host memory
    free(h_input);
    free(h_intermediate);

    return 0;
}