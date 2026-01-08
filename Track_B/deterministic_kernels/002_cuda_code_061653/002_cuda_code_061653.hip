#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

// Define a simple quad tree node structure
struct QuadTreeNode {
    int data;
    int children[4]; // Indices of the children nodes, -1 if no child
};

// Kernel to initialize the quad tree nodes
__global__ void initializeQuadTreeNodes(QuadTreeNode* nodes, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        nodes[idx].data = idx; // Assign some arbitrary data
        for (int i = 0; i < 4; ++i) {
            nodes[idx].children[i] = -1; // No children initially
        }
    }
}

// Kernel to demonstrate a simple operation on the quad tree
__global__ void processQuadTreeNodes(QuadTreeNode* nodes, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        // Example operation: set the data of the node to the sum of its children's data
        int sum = 0;
        for (int i = 0; i < 4; ++i) {
            if (nodes[idx].children[i] != -1) {
                sum += nodes[nodes[idx].children[i]].data;
            }
        }
        nodes[idx].data = sum;
    }
}

int main() {
    const int numNodes = 16384;
    const int blockSize = 256;
    const int numBlocks = (numNodes + blockSize - 1) / blockSize;

    // Allocate memory on the device
    QuadTreeNode* d_nodes;
    hipError_t err = hipMalloc(&d_nodes, numNodes * sizeof(QuadTreeNode));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory on device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize the quad tree nodes on the device
    initializeQuadTreeNodes<<<numBlocks, blockSize>>>(d_nodes, numNodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Synchronize to ensure the initialization is complete
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Example operation on the quad tree
    processQuadTreeNodes<<<numBlocks, blockSize>>>(d_nodes, numNodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Synchronize to ensure the operation is complete
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Copy the results back to the host for verification
    thrust::device_ptr<QuadTreeNode> d_ptr(d_nodes);
    thrust::host_vector<QuadTreeNode> h_nodes(d_ptr, d_ptr + numNodes);

    // Free the device memory
    hipFree(d_nodes);

    // Output the result of the first 10 nodes for verification
    std::cout << "First 10 nodes after processing:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "Node " << i << ": Data = " << h_nodes[i].data << std::endl;
    }

    return 0;
}