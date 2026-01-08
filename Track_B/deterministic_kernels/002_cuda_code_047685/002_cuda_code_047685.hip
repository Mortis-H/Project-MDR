#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

// Define a simple B-tree node structure with 64 nodes
#define NUM_NODES 64
#define NODE_SIZE 4  // Each node can have up to 4 children

struct BTreeNode {
    int keys[NODE_SIZE - 1];
    int children[NODE_SIZE];
    int n;  // Number of keys in this node
};

// Kernel to initialize B-tree nodes
__global__ void initBTreeNodes(BTreeNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        // Initialize each node with default values
        BTreeNode* node = &nodes[idx];
        node->n = 0;
        for (int i = 0; i < NODE_SIZE - 1; ++i) {
            node->keys[i] = 0;
        }
        for (int i = 0; i < NODE_SIZE; ++i) {
            node->children[i] = -1;  // -1 indicates no child
        }
    }
}

// Kernel to perform a simple operation using shared memory and cooperative groups
__global__ void bTreeOperation(BTreeNode* nodes) {
    extern __shared__ int sharedData[];

    // Define a cooperative group
    cg::thread_block cta = cg::this_thread_block();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        // Load data into shared memory
        for (int i = 0; i < NODE_SIZE - 1; ++i) {
            sharedData[threadIdx.x * (NODE_SIZE - 1) + i] = nodes[idx].keys[i];
        }
        cg::sync(cta);

        // Perform a simple operation: increment each key
        for (int i = 0; i < NODE_SIZE - 1; ++i) {
            sharedData[threadIdx.x * (NODE_SIZE - 1) + i]++;
        }
        cg::sync(cta);

        // Write data back to global memory
        for (int i = 0; i < NODE_SIZE - 1; ++i) {
            nodes[idx].keys[i] = sharedData[threadIdx.x * (NODE_SIZE - 1) + i];
        }
    }
}

int main() {
    BTreeNode* d_nodes;
    size_t size = NUM_NODES * sizeof(BTreeNode);

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_nodes, size);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory on device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize nodes on the device
    int blockSize = 256;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;
    initBTreeNodes<<<numBlocks, blockSize>>>(d_nodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initBTreeNodes kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Perform operations on nodes using shared memory and cooperative groups
    size_t sharedMemSize = blockSize * (NODE_SIZE - 1) * sizeof(int);
    bTreeOperation<<<numBlocks, blockSize, sharedMemSize>>>(d_nodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch bTreeOperation kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Copy results back to host for verification
    BTreeNode* h_nodes = new BTreeNode[NUM_NODES];
    err = hipMemcpy(h_nodes, d_nodes, size, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device to host: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        delete[] h_nodes;
        return -1;
    }

    // Verify results
    bool success = true;
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < h_nodes[i].n; ++j) {
            if (h_nodes[i].keys[j] != 1) {
                success = false;
                break;
            }
        }
    }

    if (success) {
        std::cout << "B-tree operations completed successfully." << std::endl;
    } else {
        std::cout << "B-tree operations failed." << std::endl;
    }

    // Free device memory
    hipFree(d_nodes);

    // Free host memory
    delete[] h_nodes;

    return 0;
}