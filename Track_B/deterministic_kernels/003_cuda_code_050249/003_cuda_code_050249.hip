#include <iostream>
#include <hip/hip_runtime.h>

// Define the maximum number of elements in the skip list
#define MAX_ELEMENTS 128

// Define the maximum level of the skip list
#define MAX_LEVEL 16

// Define a node in the skip list
struct SkipListNode {
    int key;
    int value;
    SkipListNode* forward[MAX_LEVEL];
};

// Kernel to initialize the skip list nodes
__global__ void initializeSkipListNodes(SkipListNode* nodes, int* levels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MAX_ELEMENTS) {
        nodes[idx].key = idx;
        nodes[idx].value = idx * 10; // Arbitrary value
        for (int i = 0; i < MAX_LEVEL; ++i) {
            nodes[idx].forward[i] = nullptr;
        }
        levels[idx] = 1; // Initialize all nodes to level 1
    }
}

// Kernel to build the skip list structure
__global__ void buildSkipList(SkipListNode* nodes, int* levels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MAX_ELEMENTS - 1) {
        int nextLevel = min(levels[idx] + 1, MAX_LEVEL);
        nodes[idx].forward[0] = &nodes[idx + 1];
        for (int level = 1; level < nextLevel; ++level) {
            if (idx + (1 << level) < MAX_ELEMENTS) {
                nodes[idx].forward[level] = &nodes[idx + (1 << level)];
            } else {
                nodes[idx].forward[level] = nullptr;
            }
        }
    }
}

// Kernel to search for a key in the skip list
__global__ void searchSkipList(SkipListNode* nodes, int* result, int searchKey) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        SkipListNode* current = &nodes[0];
        for (int level = MAX_LEVEL - 1; level >= 0; --level) {
            while (current->forward[level] != nullptr && current->forward[level]->key < searchKey) {
                current = current->forward[level];
            }
        }
        current = current->forward[0];
        if (current != nullptr && current->key == searchKey) {
            *result = current->value;
        } else {
            *result = -1; // Key not found
        }
    }
}

int main() {
    // Allocate memory for nodes and levels on the host
    SkipListNode* h_nodes = new SkipListNode[MAX_ELEMENTS];
    int* h_levels = new int[MAX_ELEMENTS];
    for (int i = 0; i < MAX_ELEMENTS; ++i) {
        h_levels[i] = 1; // Initialize all nodes to level 1
    }

    // Allocate memory for nodes and levels on the device
    SkipListNode* d_nodes;
    int* d_levels;
    int* d_result;
    hipError_t err = hipMalloc(&d_nodes, MAX_ELEMENTS * sizeof(SkipListNode));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for nodes" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_levels, MAX_ELEMENTS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for levels" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_result, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for result" << std::endl;
        return -1;
    }

    // Copy levels to device
    err = hipMemcpy(d_levels, h_levels, MAX_ELEMENTS * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy levels to device" << std::endl;
        return -1;
    }

    // Initialize nodes on the device
    int threadsPerBlock = 256;
    int blocksPerGrid = (MAX_ELEMENTS + threadsPerBlock - 1) / threadsPerBlock;
    initializeSkipListNodes<<<blocksPerGrid, threadsPerBlock>>>(d_nodes, d_levels);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Build the skip list structure on the device
    buildSkipList<<<blocksPerGrid, threadsPerBlock>>>(d_nodes, d_levels);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Search for a key in the skip list
    int searchKey = 64;
    searchSkipList<<<1, 1>>>(d_nodes, d_result, searchKey);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy the result back to the host
    int h_result;
    err = hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy result to host" << std::endl;
        return -1;
    }

    // Output the result
    std::cout << "Value for key " << searchKey << ": " << h_result << std::endl;

    // Free device memory
    hipFree(d_nodes);
    hipFree(d_levels);
    hipFree(d_result);

    // Free host memory
    delete[] h_nodes;
    delete[] h_levels;

    return 0;
}