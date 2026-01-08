#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

// Kernel to perform parallel reduction using shared memory and cooperative groups
__global__ void reduceKernel(float* d_out, const float* d_in, int n) {
    extern __shared__ float sdata[];

    // Get the thread index within the block
    int tid = threadIdx.x;

    // Load shared mem from global mem
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        sdata[tid] = d_in[i];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

// Kernel to perform final reduction on block sums
__global__ void finalReduceKernel(float* d_out, const float* d_in, int n) {
    extern __shared__ float sdata[];

    // Get the thread index within the block
    int tid = threadIdx.x;

    // Load shared mem from global mem
    int i = tid;
    if (i < n) {
        sdata[tid] = d_in[i];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) {
        d_out[0] = sdata[0];
    }
}

int main() {
    const int n = 8192;
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    // Allocate memory on the host
    float* h_in = new float[n];
    float* h_out = new float[numBlocks];

    // Initialize host data
    for (int i = 0; i < n; i++) {
        h_in[i] = 1.0f; // Example data
    }

    // Allocate memory on the device
    float* d_in;
    float* d_intermediate;
    float* d_out;
    hipError_t err = hipMalloc(&d_in, n * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for d_in: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&d_intermediate, numBlocks * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for d_intermediate: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&d_out, sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for d_out: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_in, h_in, n * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying data to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch first reduction kernel
    reduceKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_intermediate, d_in, n);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching reduceKernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch final reduction kernel
    finalReduceKernel<<<1, blockSize, blockSize * sizeof(float)>>>(d_out, d_intermediate, numBlocks);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching finalReduceKernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy result back to host
    err = hipMemcpy(h_out, d_out, sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying data back to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Output the result
    std::cout << "Reduction result: " << h_out[0] << std::endl;

    // Free device memory
    hipFree(d_in);
    hipFree(d_intermediate);
    hipFree(d_out);

    // Free host memory
    delete[] h_in;
    delete[] h_out;

    return 0;
}