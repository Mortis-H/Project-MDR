#include <iostream>
#include <hip/hip_runtime.h>


// Kernel to perform partial reduction using dynamic parallelism
__global__ void partialReductionKernel(float* d_input, float* d_partialSums, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (i < n) ? d_input[i] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        d_partialSums[blockIdx.x] = sdata[0];
    }
}

// Kernel to compute the mean of the partial sums
__global__ void computeMeanKernel(float* d_partialSums, float* d_mean, int numBlocks) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (tid < numBlocks) ? d_partialSums[tid] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        *d_mean = sdata[0] / numBlocks;
    }
}

// Kernel to compute the variance
__global__ void computeVarianceKernel(float* d_input, float mean, float* d_variance, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory and compute squared differences
    sdata[tid] = (i < n) ? (d_input[i] - mean) * (d_input[i] - mean) : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        atomicAdd(d_variance, sdata[0]);
    }
}

int main() {
    const int n = 16384;
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    float* d_input;
    float* d_partialSums;
    float* d_mean;
    float* d_variance;

    // Allocate device memory
    hipError_t err = hipMalloc(&d_input, n * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for input" << std::endl;
        return -1;
    }

    err = hipMalloc(&d_partialSums, numBlocks * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for partial sums" << std::endl;
        return -1;
    }

    err = hipMalloc(&d_mean, sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for mean" << std::endl;
        return -1;
    }

    err = hipMalloc(&d_variance, sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for variance" << std::endl;
        return -1;
    }

    // Initialize input data on the host
    float* h_input = new float[n];
    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Copy input data to the device
    err = hipMemcpy(d_input, h_input, n * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy input data to device" << std::endl;
        return -1;
    }

    // Set initial variance to 0
    float h_variance = 0.0f;
    err = hipMemcpy(d_variance, &h_variance, sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize variance on device" << std::endl;
        return -1;
    }

    // Launch partial reduction kernel
    partialReductionKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_partialSums, n);
    if (err != hipSuccess) {
        std::cerr << "Failed to launch partial reduction kernel" << std::endl;
        return -1;
    }

    // Launch compute mean kernel
    computeMeanKernel<<<1, blockSize, blockSize * sizeof(float)>>>(d_partialSums, d_mean, numBlocks);
    if (err != hipSuccess) {
        std::cerr << "Failed to launch compute mean kernel" << std::endl;
        return -1;
    }

    // Synchronize to ensure mean is computed
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize after compute mean kernel" << std::endl;
        return -1;
    }

    // Retrieve mean from device
    float h_mean;
    err = hipMemcpy(&h_mean, d_mean, sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy mean from device to host" << std::endl;
        return -1;
    }

    // Launch compute variance kernel
    computeVarianceKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, h_mean, d_variance, n);
    if (err != hipSuccess) {
        std::cerr << "Failed to launch compute variance kernel" << std::endl;
        return -1;
    }

    // Synchronize to ensure variance is computed
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize after compute variance kernel" << std::endl;
        return -1;
    }

    // Retrieve variance from device
    err = hipMemcpy(&h_variance, d_variance, sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy variance from device to host" << std::endl;
        return -1;
    }

    // Normalize variance
    h_variance /= n;

    // Output the result
    std::cout << "Computed variance: " << h_variance << std::endl;

    // Free device memory
    hipFree(d_input);
    hipFree(d_partialSums);
    hipFree(d_mean);
    hipFree(d_variance);

    // Free host memory
    delete[] h_input;

    return 0;
}