// btree_cuda.cu
// A simple CUDA program demonstrating basic operations on a B-tree with 256 nodes
// using dynamic parallelism on a Turing architecture.

#include <iostream>
#include <hip/hip_runtime.h>

#define NUM_NODES 256
#define MAX_CHILDREN 4  // Assuming a 4-way B-tree for simplicity

// Structure to represent a node in the B-tree
struct BTreeNode {
    int keys[MAX_CHILDREN - 1];
    int count;
    int children[MAX_CHILDREN];
};

// Kernel to initialize B-tree nodes
__global__ void initializeNodes(BTreeNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        nodes[idx].count = 0;
        for (int i = 0; i < MAX_CHILDREN; ++i) {
            nodes[idx].children[i] = -1;  // -1 indicates no child
        }
    }
}

// Kernel to insert a key into a B-tree node
__global__ void insertKey(BTreeNode* nodes, int nodeIdx, int key) {
    int idx = threadIdx.x;
    BTreeNode* node = &nodes[nodeIdx];
    if (idx < node->count) {
        // Find the position to insert the new key
        int i = node->count - 1;
        while (i >= 0 && node->keys[i] > key) {
            node->keys[i + 1] = node->keys[i];
            i--;
        }
        node->keys[i + 1] = key;
        node->count++;
    }
}

// Main function to demonstrate B-tree operations using CUDA
int main() {
    BTreeNode* d_nodes;
    hipError_t err;

    // Allocate memory on the device for B-tree nodes
    err = hipMalloc(&d_nodes, NUM_NODES * sizeof(BTreeNode));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for B-tree nodes (error code " << err << ")" << std::endl;
        return -1;
    }

    // Initialize nodes on the device
    initializeNodes<<<(NUM_NODES + 255) / 256, 256>>>(d_nodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initializeNodes kernel (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize after initializeNodes kernel (error code " << err << ")" << std::endl;
        return -1;
    }

    // Insert keys into the B-tree nodes
    int nodeIdx = 0;  // Inserting into the root node for simplicity
    int keysToInsert[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    int numKeys = sizeof(keysToInsert) / sizeof(keysToInsert[0]);

    for (int i = 0; i < numKeys; ++i) {
        insertKey<<<1, 1>>>(d_nodes, nodeIdx, keysToInsert[i]);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to launch insertKey kernel (error code " << err << ")" << std::endl;
            return -1;
        }
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::cerr << "Failed to synchronize after insertKey kernel (error code " << err << ")" << std::endl;
            return -1;
        }
    }

    // Free device memory
    err = hipFree(d_nodes);
    if (err != hipSuccess) {
        std::cerr << "Failed to free device memory (error code " << err << ")" << std::endl;
        return -1;
    }

    std::cout << "B-tree operations completed successfully." << std::endl;
    return 0;
}