// Include necessary headers
#include <iostream>
#include <hip/hip_runtime.h>

// Kernel to perform partial reduction
__global__ void partialReduction(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];

    // Load shared mem from global mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kernel to perform final reduction and compute average
__global__ void finalReductionAndAverage(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];

    // Load shared mem from global mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) {
        float sum = sdata[0];
        *g_odata = sum / n; // Compute average
    }
}

// Main function to demonstrate the usage of the kernels
int main(void) {
    int N = 128; // Size of the array
    float h_idata[N]; // Host input array
    float h_odata; // Host output average

    // Initialize host array
    for (int i = 0; i < N; i++) {
        h_idata[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_idata, *d_odata;
    hipError_t err = hipMalloc((void**)&d_idata, N * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for d_idata (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_odata, (N / 256 + 1) * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for d_odata (error code " << err << ")" << std::endl;
        return -1;
    }

    // Copy input data from host to device
    err = hipMemcpy(d_idata, h_idata, N * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from host to device (error code " << err << ")" << std::endl;
        return -1;
    }

    // Launch partial reduction kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    partialReduction<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_idata, d_odata, N);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed (error code " << err << ")" << std::endl;
        return -1;
    }

    // Launch final reduction and average computation kernel
    numBlocks = (numBlocks + blockSize - 1) / blockSize;
    finalReductionAndAverage<<<1, blockSize, blockSize * sizeof(float)>>>(d_odata, d_odata, numBlocks);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed (error code " << err << ")" << std::endl;
        return -1;
    }

    // Copy result from device to host
    err = hipMemcpy(&h_odata, d_odata, sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device to host (error code " << err << ")" << std::endl;
        return -1;
    }

    // Print the result
    std::cout << "Average: " << h_odata << std::endl;

    // Free device memory
    hipFree(d_idata);
    hipFree(d_odata);

    return 0;
}