// btree_cuda.cu
// A simple CUDA program demonstrating operations on a B-tree with 128 nodes.
// Optimized for persistent threads on the Hopper architecture.

#include <iostream>
#include <hip/hip_runtime.h>

// Define the maximum number of nodes in the B-tree
#define NUM_NODES 128

// Define a simple B-tree node structure
struct BTreeNode {
    int data;
    int left;
    int right;
};

// CUDA kernel to initialize B-tree nodes
__global__ void initBTreeNodes(BTreeNode* nodes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_NODES) {
        // Initialize each node with some data and child pointers
        nodes[idx].data = idx;
        nodes[idx].left = (idx * 2 + 1 < NUM_NODES) ? idx * 2 + 1 : -1;
        nodes[idx].right = (idx * 2 + 2 < NUM_NODES) ? idx * 2 + 2 : -1;
    }
}

// CUDA kernel to perform a simple operation (e.g., increment data) on each node
__global__ void operateOnNodes(BTreeNode* nodes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_NODES) {
        // Perform an operation on the node data
        nodes[idx].data += 10;
    }
}

int main() {
    // Allocate memory for the B-tree nodes on the host
    BTreeNode* h_nodes = (BTreeNode*)malloc(NUM_NODES * sizeof(BTreeNode));

    // Allocate memory for the B-tree nodes on the device
    BTreeNode* d_nodes;
    hipError_t err = hipMalloc((void**)&d_nodes, NUM_NODES * sizeof(BTreeNode));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory on device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize the grid and block dimensions
    dim3 blockSize(256);
    dim3 gridSize((NUM_NODES + blockSize.x - 1) / blockSize.x);

    // Launch the kernel to initialize the B-tree nodes
    initBTreeNodes<<<gridSize, blockSize>>>(d_nodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching initBTreeNodes kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch the kernel to perform operations on the B-tree nodes
    operateOnNodes<<<gridSize, blockSize>>>(d_nodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching operateOnNodes kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy the results back to the host
    err = hipMemcpy(h_nodes, d_nodes, NUM_NODES * sizeof(BTreeNode), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying data from device to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Print the results
    std::cout << "B-tree node data after operations:" << std::endl;
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << "Node " << i << ": data = " << h_nodes[i].data
                  << ", left = " << h_nodes[i].left << ", right = " << h_nodes[i].right << std::endl;
    }

    // Free the device memory
    hipFree(d_nodes);

    // Free the host memory
    free(h_nodes);

    return 0;
}