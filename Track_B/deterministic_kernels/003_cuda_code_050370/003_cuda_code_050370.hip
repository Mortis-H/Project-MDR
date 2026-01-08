#include <iostream>
#include <hip/hip_runtime.h>

// Define the number of particles
#define NUM_PARTICLES 1024

// Kernel to initialize the PageRank values
__global__ void initializePageRank(float *pageRank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        pageRank[idx] = 1.0f / NUM_PARTICLES;
    }
}

// Kernel to update the PageRank values
__global__ void updatePageRank(float *pageRank, float *newPageRank, int *outDegree, float d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        float sum = 0.0f;
        // Simulate the summation of contributions from other particles
        for (int i = 0; i < NUM_PARTICLES; ++i) {
            if (i != idx && outDegree[i] > 0) {
                sum += pageRank[i] / outDegree[i];
            }
        }
        newPageRank[idx] = (1.0f - d) / NUM_PARTICLES + d * sum;
    }
}

// Kernel to copy new PageRank values to the old PageRank array
__global__ void copyPageRank(float *pageRank, float *newPageRank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        pageRank[idx] = newPageRank[idx];
    }
}

int main() {
    // Allocate memory for PageRank values and outDegrees
    float *h_pageRank, *h_newPageRank;
    int *h_outDegree;
    float *d_pageRank, *d_newPageRank;
    int *d_outDegree;

    // Allocate host memory
    h_pageRank = (float *)malloc(NUM_PARTICLES * sizeof(float));
    h_newPageRank = (float *)malloc(NUM_PARTICLES * sizeof(float));
    h_outDegree = (int *)malloc(NUM_PARTICLES * sizeof(int));

    // Initialize outDegrees randomly for demonstration
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_outDegree[i] = (i % 10) + 1; // Random out-degree between 1 and 10
    }

    // Allocate device memory
    hipMalloc((void **)&d_pageRank, NUM_PARTICLES * sizeof(float));
    hipMalloc((void **)&d_newPageRank, NUM_PARTICLES * sizeof(float));
    hipMalloc((void **)&d_outDegree, NUM_PARTICLES * sizeof(int));

    // Copy outDegrees to device
    hipMemcpy(d_outDegree, h_outDegree, NUM_PARTICLES * sizeof(int), hipMemcpyHostToDevice);

    // Initialize PageRank values on the device
    int blockSize = 256;
    int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;
    initializePageRank<<<numBlocks, blockSize>>>(d_pageRank);

    // Check for errors in kernel launch
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error in initializePageRank: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Damping factor
    float d = 0.85f;

    // Perform a few iterations of PageRank
    for (int iter = 0; iter < 10; ++iter) {
        updatePageRank<<<numBlocks, blockSize>>>(d_pageRank, d_newPageRank, d_outDegree, d);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error in updatePageRank: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        copyPageRank<<<numBlocks, blockSize>>>(d_pageRank, d_newPageRank);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error in copyPageRank: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy the final PageRank values back to the host
    hipMemcpy(h_pageRank, d_pageRank, NUM_PARTICLES * sizeof(float), hipMemcpyDeviceToHost);

    // Print the first 10 PageRank values for verification
    std::cout << "First 10 PageRank values:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "PageRank[" << i << "] = " << h_pageRank[i] << std::endl;
    }

    // Free device memory
    hipFree(d_pageRank);
    hipFree(d_newPageRank);
    hipFree(d_outDegree);

    // Free host memory
    free(h_pageRank);
    free(h_newPageRank);
    free(h_outDegree);

    return 0;
}