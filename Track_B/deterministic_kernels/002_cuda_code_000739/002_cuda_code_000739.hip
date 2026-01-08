// skip_list_cuda.cu
// A simple CUDA implementation demonstrating skip list operations with 128 elements
// Optimized for instruction-level parallelism on Hopper architecture

#include <iostream>
#include <hip/hip_runtime.h>

#define NUM_ELEMENTS 128

// Kernel to initialize the skip list with random values
__global__ void initSkipList(int *skipList) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        skipList[idx] = idx; // Simple initialization for demonstration
    }
}

// Kernel to perform a basic search operation in the skip list
__global__ void searchSkipList(int *skipList, int target, int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS && skipList[idx] == target) {
        atomicExch(result, idx); // Store the index of the found element
    }
}

int main() {
    int *d_skipList, *d_result;
    int h_result = -1; // Initialize result to -1 indicating not found
    int *h_skipList = new int[NUM_ELEMENTS];

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_skipList, NUM_ELEMENTS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for skipList: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_result, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for result: " << hipGetErrorString(err) << std::endl;
        hipFree(d_skipList);
        return -1;
    }

    // Initialize the result on the device to -1
    err = hipMemcpy(d_result, &h_result, sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying result to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_skipList);
        hipFree(d_result);
        return -1;
    }

    // Initialize the skip list on the device
    int blockSize = 256;
    int numBlocks = (NUM_ELEMENTS + blockSize - 1) / blockSize;
    initSkipList<<<numBlocks, blockSize>>>(d_skipList);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching initSkipList kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_skipList);
        hipFree(d_result);
        delete[] h_skipList;
        return -1;
    }

    // Perform a search operation
    int target = 64; // Search for the element with value 64
    searchSkipList<<<numBlocks, blockSize>>>(d_skipList, target, d_result);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching searchSkipList kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_skipList);
        hipFree(d_result);
        delete[] h_skipList;
        return -1;
    }

    // Copy the result back to the host
    err = hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying result back to host: " << hipGetErrorString(err) << std::endl;
        hipFree(d_skipList);
        hipFree(d_result);
        delete[] h_skipList;
        return -1;
    }

    // Output the result
    if (h_result != -1) {
        std::cout << "Element " << target << " found at index " << h_result << std::endl;
    } else {
        std::cout << "Element " << target << " not found in the skip list." << std::endl;
    }

    // Free device memory
    hipFree(d_skipList);
    hipFree(d_result);
    delete[] h_skipList;

    return 0;
}