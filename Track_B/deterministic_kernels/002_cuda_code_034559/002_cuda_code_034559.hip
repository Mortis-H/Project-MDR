#include <iostream>
#include <hip/hip_runtime.h>

// Define the block size for the histogram kernel
#define BLOCK_SIZE 256

// Kernel to compute partial histograms
__global__ void computePartialHistograms(const int* d_input, unsigned int* d_partialHistograms, int numBins) {
    extern __shared__ unsigned int s_histogram[];

    // Initialize shared memory histogram
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        s_histogram[i] = 0;
    }
    __syncthreads();

    // Compute partial histogram for this block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 512) {
        atomicAdd(&s_histogram[d_input[idx]], 1);
    }
    __syncthreads();

    // Write partial histogram to global memory
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        atomicAdd(&d_partialHistograms[blockIdx.x * numBins + i], s_histogram[i]);
    }
}

// Kernel to reduce partial histograms to the final histogram
__global__ void reduceHistograms(unsigned int* d_partialHistograms, unsigned int* d_finalHistogram, int numBins, int numBlocks) {
    extern __shared__ unsigned int s_histogram[];

    // Initialize shared memory histogram
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        s_histogram[i] = 0;
    }
    __syncthreads();

    // Sum partial histograms into shared memory
    for (int block = 0; block < numBlocks; ++block) {
        int idx = block * numBins + threadIdx.x;
        if (threadIdx.x < numBins) {
            atomicAdd(&s_histogram[threadIdx.x], d_partialHistograms[idx]);
        }
        __syncthreads();
    }

    // Write final histogram to global memory
    if (threadIdx.x < numBins) {
        d_finalHistogram[threadIdx.x] = s_histogram[threadIdx.x];
    }
}

int main() {
    // Initialize input data
    int h_input[512];
    for (int i = 0; i < 512; ++i) {
        h_input[i] = i % 16; // Example data: 16 bins
    }

    // Allocate device memory
    int* d_input;
    unsigned int* d_partialHistograms;
    unsigned int* d_finalHistogram;
    hipError_t err;

    err = hipMalloc(&d_input, 512 * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device input memory" << std::endl;
        return -1;
    }

    err = hipMalloc(&d_partialHistograms, 32 * 16 * sizeof(unsigned int)); // 32 blocks, 16 bins
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device partial histograms memory" << std::endl;
        hipFree(d_input);
        return -1;
    }

    err = hipMalloc(&d_finalHistogram, 16 * sizeof(unsigned int)); // 16 bins
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device final histogram memory" << std::endl;
        hipFree(d_input);
        hipFree(d_partialHistograms);
        return -1;
    }

    // Copy input data to device
    err = hipMemcpy(d_input, h_input, 512 * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy input data to device" << std::endl;
        hipFree(d_input);
        hipFree(d_partialHistograms);
        hipFree(d_finalHistogram);
        return -1;
    }

    // Launch computePartialHistograms kernel
    int numBins = 16;
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(32); // 32 blocks for 512 elements

    computePartialHistograms<<<gridSize, blockSize, numBins * sizeof(unsigned int)>>>(d_input, d_partialHistograms, numBins);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch computePartialHistograms kernel" << std::endl;
        hipFree(d_input);
        hipFree(d_partialHistograms);
        hipFree(d_finalHistogram);
        return -1;
    }

    // Launch reduceHistograms kernel
    reduceHistograms<<<1, blockSize, numBins * sizeof(unsigned int)>>>(d_partialHistograms, d_finalHistogram, numBins, gridSize.x);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch reduceHistograms kernel" << std::endl;
        hipFree(d_input);
        hipFree(d_partialHistograms);
        hipFree(d_finalHistogram);
        return -1;
    }

    // Copy final histogram back to host
    unsigned int h_finalHistogram[16];
    err = hipMemcpy(h_finalHistogram, d_finalHistogram, 16 * sizeof(unsigned int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy final histogram to host" << std::endl;
        hipFree(d_input);
        hipFree(d_partialHistograms);
        hipFree(d_finalHistogram);
        return -1;
    }

    // Print final histogram
    std::cout << "Final Histogram:" << std::endl;
    for (int i = 0; i < 16; ++i) {
        std::cout << "Bin " << i << ": " << h_finalHistogram[i] << std::endl;
    }

    // Free device memory
    hipFree(d_input);
    hipFree(d_partialHistograms);
    hipFree(d_finalHistogram);

    return 0;
}