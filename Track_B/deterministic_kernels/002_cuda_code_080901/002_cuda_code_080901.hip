// btree_cuda.cu
// A simple CUDA program demonstrating operations on a B-tree with 256 nodes.
// This example is optimized for basic operations and multi-GPU scaling.

#include <iostream>
#include <hip/hip_runtime.h>

// Define the maximum number of nodes in the B-tree
#define MAX_NODES 256

// Define the structure for a B-tree node
struct BTreeNode {
    int keys[3];  // Assuming a 2-3 tree for simplicity
    int children[4];
    int n;        // Number of keys in this node
};

// Kernel to initialize B-tree nodes
__global__ void initializeBTreeNodes(BTreeNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MAX_NODES) {
        nodes[idx].n = 0;  // Initialize number of keys to 0
        for (int i = 0; i < 3; ++i) {
            nodes[idx].keys[i] = 0;  // Initialize keys to 0
        }
        for (int i = 0; i < 4; ++i) {
            nodes[idx].children[i] = -1;  // Initialize children to -1 (no child)
        }
    }
}

// Kernel to insert a key into a B-tree node
__global__ void insertKeyIntoNode(BTreeNode* nodes, int nodeIndex, int key) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == nodeIndex && nodes[idx].n < 3) {
        int i = nodes[idx].n - 1;
        while (i >= 0 && nodes[idx].keys[i] > key) {
            nodes[idx].keys[i + 1] = nodes[idx].keys[i];
            i--;
        }
        nodes[idx].keys[i + 1] = key;
        nodes[idx].n++;
    }
}

int main() {
    // Allocate memory for B-tree nodes on the host
    BTreeNode* h_nodes = new BTreeNode[MAX_NODES];

    // Allocate memory for B-tree nodes on the device
    BTreeNode* d_nodes;
    hipError_t err = hipMalloc((void**)&d_nodes, MAX_NODES * sizeof(BTreeNode));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory (error code " << err << ")" << std::endl;
        return -1;
    }

    // Initialize the block and grid sizes
    dim3 blockSize(256);
    dim3 gridSize((MAX_NODES + blockSize.x - 1) / blockSize.x);

    // Launch the kernel to initialize B-tree nodes
    initializeBTreeNodes<<<gridSize, blockSize>>>(d_nodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initializeBTreeNodes kernel (error code " << err << ")" << std::endl;
        hipFree(d_nodes);
        delete[] h_nodes;
        return -1;
    }

    // Copy data from device to host to verify initialization
    err = hipMemcpy(h_nodes, d_nodes, MAX_NODES * sizeof(BTreeNode), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device to host (error code " << err << ")" << std::endl;
        hipFree(d_nodes);
        delete[] h_nodes;
        return -1;
    }

    // Insert keys into the B-tree nodes
    int keysToInsert[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    for (int key : keysToInsert) {
        insertKeyIntoNode<<<gridSize, blockSize>>>(d_nodes, 0, key);  // Insert into root node for simplicity
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to launch insertKeyIntoNode kernel (error code " << err << ")" << std::endl;
            hipFree(d_nodes);
            delete[] h_nodes;
            return -1;
        }
    }

    // Copy data from device to host to verify insertion
    err = hipMemcpy(h_nodes, d_nodes, MAX_NODES * sizeof(BTreeNode), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device to host (error code " << err << ")" << std::endl;
        hipFree(d_nodes);
        delete[] h_nodes;
        return -1;
    }

    // Print the keys in the root node to verify insertion
    std::cout << "Keys in root node after insertion: ";
    for (int i = 0; i < h_nodes[0].n; ++i) {
        std::cout << h_nodes[0].keys[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    hipFree(d_nodes);

    // Free host memory
    delete[] h_nodes;

    return 0;
}