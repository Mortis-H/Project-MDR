#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

// Define the maximum number of elements in the skip list
#define MAX_ELEMENTS 128

// Node structure for the skip list
struct SkipListNode {
    int value;
    int next[1]; // Simplified to level 1 for demonstration
};

// Kernel to initialize the skip list nodes
__global__ void initSkipListKernel(SkipListNode* nodes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < MAX_ELEMENTS) {
        nodes[idx].value = idx; // Initialize with index value
        nodes[idx].next[0] = idx + 1; // Point to the next node
    }
}

// Kernel to perform a simple search in the skip list
__global__ void searchSkipListKernel(SkipListNode* nodes, int target, int* result) {
    cg::thread_block cta = cg::this_thread_block();
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Use cooperative group to perform a simple search
    if (idx < MAX_ELEMENTS) {
        int current = idx;
        while (current < MAX_ELEMENTS && nodes[current].value < target) {
            current = nodes[current].next[0];
        }
        if (current < MAX_ELEMENTS && nodes[current].value == target) {
            atomicMin(result, current); // Store the index of the found element
        }
    }
    cta.sync();
}

int main() {
    SkipListNode* d_nodes;
    int* d_result;
    int h_result = MAX_ELEMENTS; // Initialize to a value out of range

    // Allocate memory on the device
    hipError_t err = hipMalloc(&d_nodes, MAX_ELEMENTS * sizeof(SkipListNode));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for nodes: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc(&d_result, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for result: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Copy initial result to device
    err = hipMemcpy(d_result, &h_result, sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy result to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_result);
        return -1;
    }

    // Launch the kernel to initialize the skip list
    int threadsPerBlock = 256;
    int blocksPerGrid = (MAX_ELEMENTS + threadsPerBlock - 1) / threadsPerBlock;
    initSkipListKernel<<<blocksPerGrid, threadsPerBlock>>>(d_nodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initSkipListKernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_result);
        return -1;
    }

    // Launch the kernel to search for a target value
    int target = 64; // Example target value
    searchSkipListKernel<<<blocksPerGrid, threadsPerBlock>>>(d_nodes, target, d_result);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch searchSkipListKernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_result);
        return -1;
    }

    // Copy the result back to the host
    err = hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy result back to host: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_result);
        return -1;
    }

    // Output the result
    if (h_result < MAX_ELEMENTS) {
        std::cout << "Target " << target << " found at index " << h_result << std::endl;
    } else {
        std::cout << "Target " << target << " not found" << std::endl;
    }

    // Free device memory
    hipFree(d_nodes);
    hipFree(d_result);

    return 0;
}