#include <iostream>
#include <hip/hip_runtime.h>

// Define the maximum number of nodes in the B-tree
#define MAX_NODES 16384

// Define the order of the B-tree (number of children per node)
#define ORDER 4

// Structure to represent a node in the B-tree
struct BTreeNode {
    int keys[2 * ORDER - 1];  // An array of keys
    BTreeNode *children[2 * ORDER];  // An array of child pointers
    int n;  // Current number of keys
    bool leaf;  // Is true when node is leaf. Otherwise false
};

// Kernel to initialize B-tree nodes
__global__ void initNodes(BTreeNode *nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MAX_NODES) {
        nodes[idx].n = 0;
        nodes[idx].leaf = true;
        for (int i = 0; i < 2 * ORDER - 1; i++) {
            nodes[idx].keys[i] = 0;
        }
        for (int i = 0; i < 2 * ORDER; i++) {
            nodes[idx].children[i] = nullptr;
        }
    }
}

// Simple kernel to demonstrate insertion of keys into the B-tree
__global__ void insertKeys(BTreeNode *nodes, int *keys, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys) {
        int key = keys[idx];
        int i = nodes[0].n - 1;
        while (i >= 0 && nodes[0].keys[i] > key) {
            nodes[0].keys[i + 1] = nodes[0].keys[i];
            i--;
        }
        nodes[0].keys[i + 1] = key;
        nodes[0].n++;
    }
}

// Function to check for CUDA errors
void checkCudaError(hipError_t err, const char *msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Allocate memory for B-tree nodes on the host
    BTreeNode *h_nodes = (BTreeNode *)malloc(MAX_NODES * sizeof(BTreeNode));
    if (!h_nodes) {
        std::cerr << "Failed to allocate host memory for B-tree nodes" << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate memory for B-tree nodes on the device
    BTreeNode *d_nodes;
    checkCudaError(hipMalloc((void **)&d_nodes, MAX_NODES * sizeof(BTreeNode)), "Failed to allocate device memory for B-tree nodes");

    // Initialize the B-tree nodes on the device
    int threadsPerBlock = 256;
    int blocksPerGrid = (MAX_NODES + threadsPerBlock - 1) / threadsPerBlock;
    initNodes<<<blocksPerGrid, threadsPerBlock>>>(d_nodes);
    checkCudaError(hipGetLastError(), "Kernel launch failed");
    checkCudaError(hipDeviceSynchronize(), "Failed to synchronize device");

    // Allocate memory for keys on the host
    int numKeys = 1024;  // Number of keys to insert
    int *h_keys = (int *)malloc(numKeys * sizeof(int));
    if (!h_keys) {
        std::cerr << "Failed to allocate host memory for keys" << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize keys with some values
    for (int i = 0; i < numKeys; i++) {
        h_keys[i] = i * 2;  // Example: even numbers
    }

    // Allocate memory for keys on the device
    int *d_keys;
    checkCudaError(hipMalloc((void **)&d_keys, numKeys * sizeof(int)), "Failed to allocate device memory for keys");

    // Copy keys from host to device
    checkCudaError(hipMemcpy(d_keys, h_keys, numKeys * sizeof(int), hipMemcpyHostToDevice), "Failed to copy keys to device");

    // Insert keys into the B-tree on the device
    blocksPerGrid = (numKeys + threadsPerBlock - 1) / threadsPerBlock;
    insertKeys<<<blocksPerGrid, threadsPerBlock>>>(d_nodes, d_keys, numKeys);
    checkCudaError(hipGetLastError(), "Kernel launch failed");
    checkCudaError(hipDeviceSynchronize(), "Failed to synchronize device");

    // Free device memory
    checkCudaError(hipFree(d_nodes), "Failed to free device memory for B-tree nodes");
    checkCudaError(hipFree(d_keys), "Failed to free device memory for keys");

    // Free host memory
    free(h_nodes);
    free(h_keys);

    std::cout << "B-tree operations completed successfully" << std::endl;

    return EXIT_SUCCESS;
}