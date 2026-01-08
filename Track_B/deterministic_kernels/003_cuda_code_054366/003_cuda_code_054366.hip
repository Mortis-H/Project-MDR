#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

namespace cg = cooperative_groups;

#define MAX_NODES 64

// Kernel to initialize the parent array for each node
__global__ void initParent(int* parent) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MAX_NODES) {
        parent[idx] = idx;
    }
}

// Kernel to perform the union operation
__global__ void unionNodes(int* parent, int* edges, int numEdges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numEdges) {
        int src = edges[idx * 2];
        int dest = edges[idx * 2 + 1];
        int srcRoot = parent[src];
        int destRoot = parent[dest];
        if (srcRoot != destRoot) {
            parent[srcRoot] = destRoot;
        }
    }
}

// Kernel to find the root of a node using path compression
__global__ void findRoot(int* parent) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MAX_NODES) {
        int root = idx;
        while (parent[root] != root) {
            root = parent[root];
        }
        parent[idx] = root;
    }
}

// Main function to demonstrate the usage of the kernel
int main() {
    // Define a simple directed graph with 64 nodes and some edges
    int h_edges[] = {
        0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
        8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15,
        15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22,
        22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29,
        29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36,
        36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42, 43,
        43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 49, 50,
        50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57,
        57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63
    };
    int numEdges = sizeof(h_edges) / (2 * sizeof(int));

    // Allocate memory for edges on the device
    int* d_edges;
    hipMalloc(&d_edges, numEdges * 2 * sizeof(int));
    hipMemcpy(d_edges, h_edges, numEdges * 2 * sizeof(int), hipMemcpyHostToDevice);

    // Allocate memory for parent array on the device
    int* d_parent;
    hipMalloc(&d_parent, MAX_NODES * sizeof(int));

    // Initialize parent array
    int blockSize = 256;
    int numBlocks = (MAX_NODES + blockSize - 1) / blockSize;
    initParent<<<numBlocks, blockSize>>>(d_parent);
    hipDeviceSynchronize();

    // Perform union operations
    numBlocks = (numEdges + blockSize - 1) / blockSize;
    unionNodes<<<numBlocks, blockSize>>>(d_parent, d_edges, numEdges);
    hipDeviceSynchronize();

    // Find roots with path compression
    numBlocks = (MAX_NODES + blockSize - 1) / blockSize;
    findRoot<<<numBlocks, blockSize>>>(d_parent);
    hipDeviceSynchronize();

    // Copy results back to host
    thrust::device_ptr<int> d_parent_ptr(d_parent);
    thrust::host_vector<int> h_parent(d_parent_ptr, d_parent_ptr + MAX_NODES);

    // Print the root of each node
    std::cout << "Roots of each node:" << std::endl;
    for (int i = 0; i < MAX_NODES; ++i) {
        std::cout << "Node " << i << ": Root " << h_parent[i] << std::endl;
    }

    // Free device memory
    hipFree(d_edges);
    hipFree(d_parent);

    return 0;
}