// skip_list_cuda.cu
// A simple CUDA implementation of skip list operations with 64 elements
// using persistent threads design pattern, optimized for occupancy on Turing architecture.

#include <iostream>
#include <hip/hip_runtime.h>

#define NUM_ELEMENTS 64
#define MAX_LEVEL 4

// Node structure for skip list
struct Node {
    int key;
    int value;
    Node* forward[MAX_LEVEL];
};

// Device memory pointers
__device__ Node* d_list[NUM_ELEMENTS];
__device__ Node d_nodes[NUM_ELEMENTS];
__device__ int d_head[MAX_LEVEL];

// Kernel to initialize the skip list
__global__ void initSkipListKernel() {
    int idx = threadIdx.x;
    if (idx == 0) {
        // Initialize head pointers
        for (int i = 0; i < MAX_LEVEL; ++i) {
            d_head[i] = -1; // -1 indicates null
        }
    }
    // Initialize nodes
    d_nodes[idx].key = idx;
    d_nodes[idx].value = idx * 10; // Example value
    for (int i = 0; i < MAX_LEVEL; ++i) {
        d_nodes[idx].forward[i] = nullptr;
    }
    d_list[idx] = &d_nodes[idx];
}

// Kernel to insert a new node into the skip list
__global__ void insertNodeKernel(int key, int value) {
    Node* update[MAX_LEVEL];
    Node* x = nullptr;
    int level = MAX_LEVEL - 1;

    // Find the correct position to insert the new node
    while (level >= 0) {
        if (d_head[level] == -1) {
            update[level] = nullptr;
            level--;
        } else {
            x = d_list[d_head[level]];
            while (x->forward[level] != nullptr && x->forward[level]->key < key) {
                x = x->forward[level];
            }
            update[level] = x;
            level--;
        }
    }

    // Prepare the new node
    Node* newNode = &d_nodes[key];
    newNode->key = key;
    newNode->value = value;

    // Insert the new node
    for (level = 0; level < MAX_LEVEL; ++level) {
        if (update[level] == nullptr) {
            newNode->forward[level] = d_list[d_head[level]];
            d_head[level] = key;
        } else {
            newNode->forward[level] = update[level]->forward[level];
            update[level]->forward[level] = newNode;
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

    // Launch kernel to initialize the skip list
    initSkipListKernel<<<1, NUM_ELEMENTS>>>();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initSkipListKernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Synchronize to ensure initialization is complete
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Insert nodes into the skip list
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        insertNodeKernel<<<1, 1>>>(i, i * 10);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to launch insertNodeKernel for key " << i << ": " << hipGetErrorString(err) << std::endl;
            return -1;
        }
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::cerr << "Failed to synchronize device after insert for key " << i << ": " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    std::cout << "Skip list initialized and populated successfully." << std::endl;

    // Clean up
    hipDeviceReset();
    return 0;
}