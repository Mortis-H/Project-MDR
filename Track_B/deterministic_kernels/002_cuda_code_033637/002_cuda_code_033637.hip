#include <iostream>
#include <hip/hip_runtime.h>


#define BLOCK_SIZE 256
#define NUM_NODES 4096

// Define a simple B-tree node structure
struct BTreeNode {
    int keys[3]; // Assuming a 2-3 tree for simplicity
    int children[4];
    int numKeys;
};

// Kernel to initialize B-tree nodes
__global__ void initBTreeNodes(BTreeNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        nodes[idx].numKeys = 0;
        for (int i = 0; i < 3; ++i) {
            nodes[idx].keys[i] = 0;
            nodes[idx].children[i] = -1;
        }
        nodes[idx].children[3] = -1;
    }
}

// Kernel to insert a key into a B-tree node
__global__ void insertKey(BTreeNode* nodes, int nodeIdx, int key) {
    int idx = threadIdx.x;
    __shared__ int sharedKeys[3 * BLOCK_SIZE];
    __shared__ int sharedChildren[4 * BLOCK_SIZE];

    // Load node data into shared memory to avoid bank conflicts
    if (idx < 3) {
        sharedKeys[threadIdx.x + threadIdx.y * 3] = nodes[nodeIdx].keys[idx];
    }
    if (idx < 4) {
        sharedChildren[threadIdx.x + threadIdx.y * 4] = nodes[nodeIdx].children[idx];
    }
    __syncthreads();

    // Insert key into shared memory array
    int insertPos = 0;
    while (insertPos < nodes[nodeIdx].numKeys && sharedKeys[insertPos] < key) {
        ++insertPos;
    }
    for (int i = nodes[nodeIdx].numKeys; i > insertPos; --i) {
        sharedKeys[i] = sharedKeys[i - 1];
        sharedChildren[i + 1] = sharedChildren[i];
    }
    sharedKeys[insertPos] = key;
    sharedChildren[insertPos + 1] = -1;
    nodes[nodeIdx].numKeys++;

    // Write back to global memory
    if (idx < 3) {
        nodes[nodeIdx].keys[idx] = sharedKeys[idx];
    }
    if (idx < 4) {
        nodes[nodeIdx].children[idx] = sharedChildren[idx];
    }
}

int main() {
    BTreeNode* d_nodes;
    hipError_t err;

    // Allocate memory for B-tree nodes on the device
    err = hipMalloc(&d_nodes, NUM_NODES * sizeof(BTreeNode));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory on device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize B-tree nodes on the device
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((NUM_NODES + blockSize.x - 1) / blockSize.x);
    initBTreeNodes<<<gridSize, blockSize>>>(d_nodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching initBTreeNodes kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Insert keys into B-tree nodes
    int keysToInsert[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    int numKeys = sizeof(keysToInsert) / sizeof(keysToInsert[0]);
    for (int i = 0; i < numKeys; ++i) {
        insertKey<<<1, BLOCK_SIZE>>>(d_nodes, 0, keysToInsert[i]);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error launching insertKey kernel: " << hipGetErrorString(err) << std::endl;
            hipFree(d_nodes);
            return -1;
        }
    }

    // Synchronize to ensure all operations are complete
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Error during device synchronization: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Free device memory
    hipFree(d_nodes);

    std::cout << "B-tree operations completed successfully." << std::endl;
    return 0;
}