#include <iostream>
#include <hip/hip_runtime.h>


#define NUM_NODES 1024
#define BLOCK_SIZE 256

// Structure to represent a node in the B-tree
struct BTreeNode {
    int keys[4]; // Assuming a 4-way B-tree for simplicity
    int numKeys;
    int children[5]; // Pointers to child nodes
};

// Kernel to initialize B-tree nodes
__global__ void initializeNodes(BTreeNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        nodes[idx].numKeys = 0;
        for (int i = 0; i < 4; i++) {
            nodes[idx].keys[i] = 0;
        }
        for (int i = 0; i < 5; i++) {
            nodes[idx].children[i] = -1; // -1 indicates no child
        }
    }
}

// Kernel to insert a key into a B-tree node
__global__ void insertKey(BTreeNode* nodes, int nodeIdx, int key) {
    int idx = threadIdx.x;
    __shared__ BTreeNode sharedNode;
    if (idx == 0) {
        sharedNode = nodes[nodeIdx];
    }
    __syncthreads();

    if (idx < sharedNode.numKeys) {
        // Find the correct position to insert the key
        int i = sharedNode.numKeys - 1;
        while (i >= 0 && sharedNode.keys[i] > key) {
            sharedNode.keys[i + 1] = sharedNode.keys[i];
            i--;
        }
        sharedNode.keys[i + 1] = key;
        sharedNode.numKeys++;
    }
    __syncthreads();

    if (idx == 0) {
        nodes[nodeIdx] = sharedNode;
    }
}

// Kernel to split a B-tree node
__global__ void splitNode(BTreeNode* nodes, int parentIdx, int childIdx) {
    int idx = threadIdx.x;
    __shared__ BTreeNode parent, child, newChild;
    if (idx == 0) {
        parent = nodes[parentIdx];
        child = nodes[childIdx];
        newChild.numKeys = 2;
    }
    __syncthreads();

    if (idx < 2) {
        newChild.keys[idx] = child.keys[idx + 3];
        newChild.children[idx] = child.children[idx + 3];
    }
    if (idx == 0) {
        newChild.children[2] = child.children[5];
        child.numKeys = 2;
        parent.keys[parent.numKeys] = child.keys[2];
        parent.children[parent.numKeys + 1] = childIdx + 1;
        parent.numKeys++;
    }
    __syncthreads();

    if (idx == 0) {
        nodes[parentIdx] = parent;
        nodes[childIdx] = child;
        nodes[childIdx + 1] = newChild;
    }
}

// Main function to demonstrate the usage of the B-tree operations
int main() {
    BTreeNode* d_nodes;
    hipError_t err;

    // Allocate memory on the device
    err = hipMalloc((void**)&d_nodes, NUM_NODES * sizeof(BTreeNode));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory on device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize nodes on the device
    initializeNodes<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_nodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching initializeNodes kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    hipDeviceSynchronize();

    // Insert keys into the B-tree
    for (int key = 1; key <= 20; key++) {
        insertKey<<<1, BLOCK_SIZE>>>(d_nodes, 0, key);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error launching insertKey kernel: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
        hipDeviceSynchronize();

        // Split the root node if necessary
        BTreeNode h_node;
        hipMemcpy(&h_node, d_nodes, sizeof(BTreeNode), hipMemcpyDeviceToHost);
        if (h_node.numKeys == 4) {
            splitNode<<<1, BLOCK_SIZE>>>(d_nodes, 0, 0);
            err = hipGetLastError();
            if (err != hipSuccess) {
                std::cerr << "Error launching splitNode kernel: " << hipGetErrorString(err) << std::endl;
                return -1;
            }
            hipDeviceSynchronize();
        }
    }

    // Free device memory
    hipFree(d_nodes);

    std::cout << "B-tree operations completed successfully." << std::endl;
    return 0;
}