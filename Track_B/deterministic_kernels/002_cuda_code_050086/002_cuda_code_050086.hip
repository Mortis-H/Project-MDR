#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

#define NUM_ELEMENTS 8192
#define BLOCK_SIZE 256

// Structure to represent a node in the skip list
struct SkipListNode {
    int value;
    int level;
    SkipListNode* next[1]; // We will allocate memory for this array dynamically
};

// Kernel to initialize skip list nodes
__global__ void initSkipListNodes(SkipListNode* nodes, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        nodes[idx].value = idx; // Assign a simple value for demonstration
        nodes[idx].level = 0;   // Initialize level to 0
        for (int i = 0; i < 1; ++i) { // Only one level for simplicity
            nodes[idx].next[i] = nullptr;
        }
    }
}

// Kernel to perform a simple operation on skip list nodes using cooperative groups
__global__ void processSkipListNodes(SkipListNode* nodes, int numElements) {
    cg::thread_block cta = cg::this_thread_block();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements) {
        // Example operation: increment the value of every node
        atomicAdd(&nodes[idx].value, 1);
    }

    // Synchronize the block to ensure all nodes are processed
    cta.sync();
}

int main() {
    SkipListNode* d_nodes;
    size_t size = NUM_ELEMENTS * sizeof(SkipListNode);

    // Allocate memory on the device
    hipError_t err = hipMalloc(&d_nodes, size);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize nodes on the device
    int numBlocks = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initSkipListNodes<<<numBlocks, BLOCK_SIZE>>>(d_nodes, NUM_ELEMENTS);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Process nodes using cooperative groups
    processSkipListNodes<<<numBlocks, BLOCK_SIZE>>>(d_nodes, NUM_ELEMENTS);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Copy results back to host for verification
    SkipListNode* h_nodes = new SkipListNode[NUM_ELEMENTS];
    err = hipMemcpy(h_nodes, d_nodes, size, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy memory from device to host: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        delete[] h_nodes;
        return -1;
    }

    // Verify results
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        if (h_nodes[i].value != i + 1) {
            std::cerr << "Mismatch at index " << i << ": expected " << i + 1 << ", got " << h_nodes[i].value << std::endl;
            hipFree(d_nodes);
            delete[] h_nodes;
            return -1;
        }
    }

    // Clean up
    hipFree(d_nodes);
    delete[] h_nodes;

    std::cout << "All elements processed correctly!" << std::endl;
    return 0;
}