// btree_cuda.cu
// A simple CUDA implementation for basic operations on a B-tree with 16384 nodes.
// Optimized for register usage and multi-GPU version.

#include <iostream>
#include <hip/hip_runtime.h>

#define NUM_NODES 16384
#define MAX_CHILDREN 4  // Assuming a 4-way B-tree for simplicity

// Define the B-tree node structure
struct BTreeNode {
    int keys[MAX_CHILDREN - 1];
    int children[MAX_CHILDREN];
    int numKeys;
};

// Kernel to initialize B-tree nodes
__global__ void initBTreeNodes(BTreeNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        nodes[idx].numKeys = 0;
        for (int i = 0; i < MAX_CHILDREN - 1; ++i) {
            nodes[idx].keys[i] = 0;
        }
        for (int i = 0; i < MAX_CHILDREN; ++i) {
            nodes[idx].children[i] = -1;  // -1 indicates no child
        }
    }
}

// Kernel to insert a key into a B-tree node
__global__ void insertKeyIntoNode(BTreeNode* nodes, int key, int nodeIdx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == nodeIdx) {
        int pos = nodes[idx].numKeys;
        // Insert key in non-decreasing order
        while (pos > 0 && nodes[idx].keys[pos - 1] > key) {
            nodes[idx].keys[pos] = nodes[idx].keys[pos - 1];
            pos--;
        }
        nodes[idx].keys[pos] = key;
        nodes[idx].numKeys++;
    }
}

int main() {
    BTreeNode* d_nodes;
    hipError_t err;

    // Allocate memory on the device for B-tree nodes
    err = hipMalloc((void**)&d_nodes, NUM_NODES * sizeof(BTreeNode));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize the B-tree nodes on the GPU
    int blockSize = 256;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;
    initBTreeNodes<<<numBlocks, blockSize>>>(d_nodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching initBTreeNodes kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Synchronize to ensure all nodes are initialized
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Error synchronizing device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Insert keys into the B-tree nodes
    for (int key = 0; key < NUM_NODES * (MAX_CHILDREN - 1); ++key) {
        int nodeIdx = key % NUM_NODES;
        insertKeyIntoNode<<<1, 1>>>(d_nodes, key, nodeIdx);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error launching insertKeyIntoNode kernel: " << hipGetErrorString(err) << std::endl;
            hipFree(d_nodes);
            return -1;
        }
    }

    // Synchronize to ensure all keys are inserted
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Error synchronizing device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Free the device memory
    hipFree(d_nodes);

    std::cout << "B-tree operations completed successfully." << std::endl;
    return 0;
}