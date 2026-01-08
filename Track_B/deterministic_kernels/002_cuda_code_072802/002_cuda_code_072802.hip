// skip_list_cuda.cu
// A basic CUDA implementation demonstrating skip list operations on 64 elements.
// Optimized for NVIDIA Hopper architecture with focus on divergence reduction.

#include <iostream>
#include <hip/hip_runtime.h>

#define N 64  // Number of elements in the skip list

// Kernel to initialize the skip list with sorted values
__global__ void initializeSkipList(int *skipList) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        skipList[idx] = idx;  // Simple initialization with sorted values
    }
}

// Kernel to perform a search operation in the skip list
__global__ void searchSkipList(int *skipList, int target, int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (skipList[idx] == target) {
            atomicExch(result, idx);  // Use atomic exchange to set the result
        }
    }
}

int main() {
    int *d_skipList, *d_result;
    int h_result = -1;  // Initialize result as -1 (not found)
    int h_skipList[N];

    // Allocate memory on the device
    hipError_t err = hipMalloc((void **)&d_skipList, N * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for skip list: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_result, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for result: " << hipGetErrorString(err) << std::endl;
        hipFree(d_skipList);
        return -1;
    }

    // Initialize device result
    err = hipMemcpy(d_result, &h_result, sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying result to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_skipList);
        hipFree(d_result);
        return -1;
    }

    // Initialize skip list on the device
    int blockSize = 16;
    int numBlocks = (N + blockSize - 1) / blockSize;
    initializeSkipList<<<numBlocks, blockSize>>>(d_skipList);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching initializeSkipList kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_skipList);
        hipFree(d_result);
        return -1;
    }

    // Perform search on the device
    int target = 32;  // Example target value to search for
    searchSkipList<<<numBlocks, blockSize>>>(d_skipList, target, d_result);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching searchSkipList kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_skipList);
        hipFree(d_result);
        return -1;
    }

    // Copy result back to host
    err = hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying result back to host: " << hipGetErrorString(err) << std::endl;
        hipFree(d_skipList);
        hipFree(d_result);
        return -1;
    }

    // Output the result
    if (h_result != -1) {
        std::cout << "Target " << target << " found at index " << h_result << std::endl;
    } else {
        std::cout << "Target " << target << " not found in the skip list." << std::endl;
    }

    // Free device memory
    hipFree(d_skipList);
    hipFree(d_result);

    return 0;
}