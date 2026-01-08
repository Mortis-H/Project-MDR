#include <iostream>
#include <hip/hip_runtime.h>

// Define the maximum number of nodes in the B-tree
#define MAX_NODES 1024

// Define the maximum number of keys in a node (simplified for demonstration)
#define MAX_KEYS 3

// Define the structure for a node in the B-tree
struct BTreeNode {
    int keys[MAX_KEYS];  // Keys stored in the node
    int count;           // Number of keys in the node
};

// CUDA kernel to initialize B-tree nodes
__global__ void initializeNodes(BTreeNode *nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MAX_NODES) {
        // Initialize each node with zero keys
        for (int i = 0; i < MAX_KEYS; ++i) {
            nodes[idx].keys[i] = 0;
        }
        nodes[idx].count = 0;
    }
}

// CUDA kernel to insert a key into a B-tree node
__global__ void insertKey(BTreeNode *nodes, int nodeIdx, int key) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {  // Only the first thread in the block performs the insertion
        BTreeNode *node = &nodes[nodeIdx];
        if (node->count < MAX_KEYS) {
            // Find the correct position to insert the key
            int i = node->count - 1;
            while (i >= 0 && node->keys[i] > key) {
                node->keys[i + 1] = node->keys[i];
                i--;
            }
            node->keys[i + 1] = key;
            node->count++;
        }
    }
}

int main() {
    // Allocate memory for B-tree nodes on the host
    BTreeNode *h_nodes = new BTreeNode[MAX_NODES];

    // Allocate memory for B-tree nodes on the device
    BTreeNode *d_nodes;
    hipError_t err = hipMalloc((void **)&d_nodes, MAX_NODES * sizeof(BTreeNode));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory on device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize nodes on the device
    int blockSize = 256;
    int numBlocks = (MAX_NODES + blockSize - 1) / blockSize;
    initializeNodes<<<numBlocks, blockSize>>>(d_nodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy initialized nodes back to the host for verification
    err = hipMemcpy(h_nodes, d_nodes, MAX_NODES * sizeof(BTreeNode), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying memory from device to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Insert keys into the B-tree nodes
    for (int i = 0; i < MAX_NODES; ++i) {
        insertKey<<<1, 1>>>(d_nodes, i, i * 2);  // Insert even numbers as keys
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error launching insert kernel: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy nodes back to the host for verification
    err = hipMemcpy(h_nodes, d_nodes, MAX_NODES * sizeof(BTreeNode), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying memory from device to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Verify the first few nodes
    for (int i = 0; i < 5; ++i) {
        std::cout << "Node " << i << ": ";
        for (int j = 0; j < h_nodes[i].count; ++j) {
            std::cout << h_nodes[i].keys[j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    hipFree(d_nodes);

    // Free host memory
    delete[] h_nodes;

    return 0;
}