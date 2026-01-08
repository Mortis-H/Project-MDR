#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define the maximum level of the skip list
#define MAX_LEVEL 16

// Node structure for the skip list
struct SkipListNode {
    int value;
    int level;
    SkipListNode* forward[MAX_LEVEL];
};

// Device memory for the skip list nodes
__device__ SkipListNode* d_skipList;

// Kernel to initialize the skip list
__global__ void initSkipList(SkipListNode* list, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        list[idx].value = idx;
        list[idx].level = 0;
        for (int i = 0; i < MAX_LEVEL; i++) {
            list[idx].forward[i] = nullptr;
        }
    }
}

// Warp-level function to insert a node into the skip list
__device__ void warpInsert(SkipListNode* list, int value, int numElements) {
    int laneId = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    int idx = warpId * warpSize + laneId;

    if (idx >= numElements) return;

    // Create a new node
    SkipListNode* newNode = &list[idx];
    newNode->value = value;
    newNode->level = 1;

    // Find the correct position to insert the new node
    SkipListNode* update[MAX_LEVEL];
    SkipListNode* current = &list[0];

    for (int i = MAX_LEVEL - 1; i >= 0; i--) {
        while (current->forward[i] != nullptr && current->forward[i]->value < value) {
            current = current->forward[i];
        }
        update[i] = current;
    }

    // Insert the new node
    for (int i = 0; i < newNode->level; i++) {
        newNode->forward[i] = update[i]->forward[i];
        update[i]->forward[i] = newNode;
    }
}

// Kernel to insert nodes into the skip list
__global__ void insertSkipList(SkipListNode* list, int numElements) {
    // Each warp handles a block of elements
    warpInsert(list, blockIdx.x * blockDim.x + threadIdx.x, numElements);
}

int main() {
    const int numElements = 1024;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory for the skip list nodes on the device
    hipError_t err = hipMalloc((void**)&d_skipList, numElements * sizeof(SkipListNode));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize the skip list
    initSkipList<<<blocksPerGrid, threadsPerBlock>>>(d_skipList, numElements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initSkipList kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_skipList);
        return -1;
    }

    // Insert elements into the skip list
    insertSkipList<<<blocksPerGrid, threadsPerBlock>>>(d_skipList, numElements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch insertSkipList kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_skipList);
        return -1;
    }

    // Free the device memory
    hipFree(d_skipList);

    std::cout << "Skip list operations completed successfully." << std::endl;
    return 0;
}