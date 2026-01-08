// btree_cuda.cu
#include <iostream>
#include <hip/hip_runtime.h>


// Define the maximum number of nodes in the B-tree
#define MAX_NODES 1024

// Define the maximum degree (number of children) for each node
#define MAX_DEGREE 16

// Structure to represent a node in the B-tree
struct BTreeNode {
    int n; // Current number of keys in this node
    int keys[MAX_DEGREE - 1]; // An array of keys
    BTreeNode* C[MAX_DEGREE]; // An array of child pointers
    bool leaf; // Is true when node is leaf. Otherwise false
};

// Device-side array to hold the B-tree nodes
__device__ BTreeNode d_nodes[MAX_NODES];

// Function to initialize the B-tree nodes on the device
__global__ void initBTreeNodes() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MAX_NODES) {
        d_nodes[idx].n = 0;
        d_nodes[idx].leaf = true;
        for (int i = 0; i < MAX_DEGREE - 1; ++i) {
            d_nodes[idx].keys[i] = 0;
        }
        for (int i = 0; i < MAX_DEGREE; ++i) {
            d_nodes[idx].C[i] = nullptr;
        }
    }
}

// Function to perform a warp-level operation on the B-tree nodes
// This example demonstrates a simple operation where we increment each key by 1
__global__ void warpLevelOperation() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MAX_NODES) {
        int lane_id = threadIdx.x % warpSize; // Get the lane ID within the warp
        int warp_id = threadIdx.x / warpSize; // Get the warp ID within the block

        // Each warp processes one node
        int node_id = warp_id + blockIdx.x * (blockDim.x / warpSize);
        if (node_id < MAX_NODES) {
            // Use shuffle to share data within the warp
            for (int i = lane_id; i < d_nodes[node_id].n; i += warpSize) {
                d_nodes[node_id].keys[i] += 1;
            }
        }
    }
}

int main() {
    // Initialize CUDA device
    hipError_t err = hipSetDevice(0);
    if (err != hipSuccess) {
        std::cerr << "Failed to set CUDA device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch kernel to initialize B-tree nodes
    int threadsPerBlock = 256;
    int blocksPerGrid = (MAX_NODES + threadsPerBlock - 1) / threadsPerBlock;
    initBTreeNodes<<<blocksPerGrid, threadsPerBlock>>>();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Synchronize to ensure all nodes are initialized
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Device synchronization failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch kernel to perform warp-level operation on B-tree nodes
    warpLevelOperation<<<blocksPerGrid, threadsPerBlock>>>();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Synchronize to ensure all operations are complete
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Device synchronization failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Print the first 10 nodes to verify the operation (host-side)
    BTreeNode h_nodes[MAX_NODES];
    err = hipMemcpy(h_nodes, d_nodes, sizeof(BTreeNode) * MAX_NODES, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Memory copy failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "First 10 nodes after operation:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "Node " << i << ": ";
        for (int j = 0; j < h_nodes[i].n; ++j) {
            std::cout << h_nodes[i].keys[j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}