#include <iostream>
#include <hip/hip_runtime.h>


#define NUM_ELEMENTS 64

// Define a simple skip list node structure
struct SkipListNode {
    int key;
    int value;
    SkipListNode* next[1]; // Simplified for 1 level skip list
};

// Kernel to initialize skip list nodes
__global__ void initSkipListNodes(SkipListNode* d_nodes, int* d_keys, int* d_values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        d_nodes[idx].key = d_keys[idx];
        d_nodes[idx].value = d_values[idx];
        d_nodes[idx].next[0] = nullptr; // Initialize next pointer to null
    }
}

// Kernel to build skip list
__global__ void buildSkipList(SkipListNode* d_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS - 1) {
        d_nodes[idx].next[0] = &d_nodes[idx + 1];
    }
}

// Kernel to search for a key in the skip list
__global__ void searchSkipList(SkipListNode* d_nodes, int searchKey, int* d_result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        SkipListNode* current = d_nodes;
        while (current != nullptr && current->key < searchKey) {
            current = current->next[0];
        }
        if (current != nullptr && current->key == searchKey) {
            *d_result = current->value;
        } else {
            *d_result = -1; // Key not found
        }
    }
}

int main() {
    // Host data
    int h_keys[NUM_ELEMENTS];
    int h_values[NUM_ELEMENTS];
    int h_result = -1;

    // Initialize host data
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        h_keys[i] = i * 2; // Even keys
        h_values[i] = i;   // Values are indices
    }

    // Device data
    SkipListNode* d_nodes;
    int* d_keys;
    int* d_values;
    int* d_result;

    // Allocate device memory
    hipError_t err = hipMalloc(&d_nodes, NUM_ELEMENTS * sizeof(SkipListNode));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for nodes" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_keys, NUM_ELEMENTS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for keys" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_values, NUM_ELEMENTS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for values" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_result, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for result" << std::endl;
        return -1;
    }

    // Asynchronous memory copy from host to device
    hipMemcpyAsync(d_keys, h_keys, NUM_ELEMENTS * sizeof(int), hipMemcpyHostToDevice, 0);
    hipMemcpyAsync(d_values, h_values, NUM_ELEMENTS * sizeof(int), hipMemcpyHostToDevice, 0);

    // Initialize skip list nodes
    dim3 threadsPerBlock(32);
    dim3 numBlocks((NUM_ELEMENTS + threadsPerBlock.x - 1) / threadsPerBlock.x);
    initSkipListNodes<<<numBlocks, threadsPerBlock>>>(d_nodes, d_keys, d_values);
    hipDeviceSynchronize();

    // Build skip list
    buildSkipList<<<numBlocks, threadsPerBlock>>>(d_nodes);
    hipDeviceSynchronize();

    // Search for a key in the skip list
    int searchKey = 10; // Example search key
    searchSkipList<<<1, 1>>>(d_nodes, searchKey, d_result);
    hipDeviceSynchronize();

    // Asynchronous memory copy from device to host
    hipMemcpyAsync(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost, 0);
    hipDeviceSynchronize();

    // Output result
    std::cout << "Search result for key " << searchKey << ": " << h_result << std::endl;

    // Free device memory
    hipFree(d_nodes);
    hipFree(d_keys);
    hipFree(d_values);
    hipFree(d_result);

    return 0;
}