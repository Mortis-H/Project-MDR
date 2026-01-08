#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

// Kernel to perform parallel reduction using cooperative groups
__global__ void reduce_stddev(float *data, float *result, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (i < n) ? data[i] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

// Kernel to compute the sum of squares
__global__ void sum_of_squares(float *data, float *mean, float *result, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    float value = (i < n) ? data[i] : 0.0f;
    sdata[tid] = value * value - 2 * value * (*mean) + (*mean) * (*mean);
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int n = 2048;
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    float *d_data, *d_intermediate, *d_mean, *d_variance;
    float h_data[n], h_mean, h_variance;

    // Initialize host data
    for (int i = 0; i < n; ++i) {
        h_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    hipError_t err = hipMalloc(&d_data, n * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate d_data (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_intermediate, numBlocks * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate d_intermediate (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_mean, sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate d_mean (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_variance, sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate d_variance (error code " << err << ")" << std::endl;
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_data, h_data, n * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from host to device (error code " << err << ")" << std::endl;
        return -1;
    }

    // First reduction to compute sum
    reduce_stddev<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_data, d_intermediate, n);
    hipDeviceSynchronize();

    // Sum the results from each block to get the total sum
    float h_intermediate[numBlocks];
    err = hipMemcpy(h_intermediate, d_intermediate, numBlocks * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy intermediate results from device to host (error code " << err << ")" << std::endl;
        return -1;
    }
    float total_sum = 0.0f;
    for (int i = 0; i < numBlocks; ++i) {
        total_sum += h_intermediate[i];
    }

    // Compute mean
    h_mean = total_sum / n;

    // Copy mean to device
    err = hipMemcpy(d_mean, &h_mean, sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy mean from host to device (error code " << err << ")" << std::endl;
        return -1;
    }

    // Second reduction to compute sum of squares
    sum_of_squares<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_data, d_mean, d_intermediate, n);
    hipDeviceSynchronize();

    // Sum the results from each block to get the total sum of squares
    err = hipMemcpy(h_intermediate, d_intermediate, numBlocks * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy intermediate results from device to host (error code " << err << ")" << std::endl;
        return -1;
    }
    float total_sum_of_squares = 0.0f;
    for (int i = 0; i < numBlocks; ++i) {
        total_sum_of_squares += h_intermediate[i];
    }

    // Compute variance and then standard deviation
    h_variance = total_sum_of_squares / n;
    float h_stddev = sqrt(h_variance);

    // Output results
    std::cout << "Mean: " << h_mean << std::endl;
    std::cout << "Variance: " << h_variance << std::endl;
    std::cout << "Standard Deviation: " << h_stddev << std::endl;

    // Free device memory
    hipFree(d_data);
    hipFree(d_intermediate);
    hipFree(d_mean);
    hipFree(d_variance);

    return 0;
}