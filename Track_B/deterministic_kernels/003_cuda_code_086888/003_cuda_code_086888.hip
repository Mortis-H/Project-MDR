#include <iostream>
#include <hip/hip_runtime.h>

#define NUM_NODES 4096
#define MAX_CHILDREN 4

// Node structure for the B-tree
struct BTreeNode {
    int keys[MAX_CHILDREN - 1];
    int numKeys;
    int children[MAX_CHILDREN];
    bool isLeaf;
};

// Kernel to initialize the B-tree nodes
__global__ void initBTreeNodes(BTreeNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        nodes[idx].numKeys = 0;
        nodes[idx].isLeaf = true;
        for (int i = 0; i < MAX_CHILDREN; ++i) {
            nodes[idx].children[i] = -1; // Initialize children to invalid index
        }
    }
}

// Kernel to insert keys into the B-tree
__global__ void insertKey(BTreeNode* nodes, int* keys, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys) {
        int key = keys[idx];
        int currentNode = 0;

        // Traverse the tree to find the correct leaf node
        while (!nodes[currentNode].isLeaf) {
            int i = 0;
            while (i < nodes[currentNode].numKeys && key > nodes[currentNode].keys[i]) {
                ++i;
            }
            currentNode = nodes[currentNode].children[i];
        }

        // Insert the key into the leaf node
        int i = nodes[currentNode].numKeys - 1;
        while (i >= 0 && nodes[currentNode].keys[i] > key) {
            nodes[currentNode].keys[i + 1] = nodes[currentNode].keys[i];
            --i;
        }
        nodes[currentNode].keys[i + 1] = key;
        nodes[currentNode].numKeys++;
    }
}

// Kernel to split a full node
__global__ void splitNode(BTreeNode* nodes, int parentIndex, int childIndex) {
    BTreeNode* child = &nodes[childIndex];
    BTreeNode* parent = &nodes[parentIndex];

    // Create a new node which is going to store (t-1) keys of child
    BTreeNode* newNode = &nodes[NUM_NODES - 1 - blockIdx.x * blockDim.x - threadIdx.x];
    newNode->isLeaf = child->isLeaf;
    newNode->numKeys = MAX_CHILDREN / 2 - 1;

    // Copy the last (t-1) keys of child to newNode
    for (int j = 0; j < MAX_CHILDREN / 2 - 1; ++j) {
        newNode->keys[j] = child->keys[j + MAX_CHILDREN / 2];
    }

    // Copy the last t children of child to newNode
    if (!child->isLeaf) {
        for (int j = 0; j < MAX_CHILDREN / 2; ++j) {
            newNode->children[j] = child->children[j + MAX_CHILDREN / 2];
        }
    }

    // Reduce the number of keys in child
    child->numKeys = MAX_CHILDREN / 2 - 1;

    // Since this node is going to have a new child, create space of new child
    for (int j = parent->numKeys; j >= 0; --j) {
        parent->children[j + 1] = parent->children[j];
    }

    // Link the new child to this node
    parent->children[parent->numKeys + 1] = NUM_NODES - 1 - blockIdx.x * blockDim.x - threadIdx.x;

    // A key of child will move to this node. Find the location of new key and move all greater keys one space ahead
    for (int j = parent->numKeys - 1; j >= 0 && child->keys[MAX_CHILDREN / 2] > parent->keys[j]; --j) {
        parent->keys[j + 1] = parent->keys[j];
    }

    // Copy the middle key of child to this node
    parent->keys[parent->numKeys] = child->keys[MAX_CHILDREN / 2];
    parent->numKeys = parent->numKeys + 1;
}

// Main function to demonstrate the usage of the B-tree kernels
int main() {
    // Allocate memory for the B-tree nodes
    BTreeNode* d_nodes;
    hipMalloc(&d_nodes, NUM_NODES * sizeof(BTreeNode));

    // Initialize the B-tree nodes
    int blockSize = 256;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;
    initBTreeNodes<<<numBlocks, blockSize>>>(d_nodes);
    hipDeviceSynchronize();
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error in initBTreeNodes: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Generate some keys to insert into the B-tree
    int* h_keys = new int[NUM_NODES];
    for (int i = 0; i < NUM_NODES; ++i) {
        h_keys[i] = i;
    }

    // Allocate memory for the keys on the device
    int* d_keys;
    hipMalloc(&d_keys, NUM_NODES * sizeof(int));
    hipMemcpy(d_keys, h_keys, NUM_NODES * sizeof(int), hipMemcpyHostToDevice);

    // Insert keys into the B-tree
    insertKey<<<numBlocks, blockSize>>>(d_nodes, d_keys, NUM_NODES);
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error in insertKey: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free allocated memory
    hipFree(d_nodes);
    hipFree(d_keys);
    delete[] h_keys;

    std::cout << "B-tree operations completed successfully." << std::endl;
    return 0;
}