#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

// Define the maximum number of nodes in the B-tree
#define MAX_NODES 32

// Structure to represent a node in the B-tree
struct BTreeNode {
    int keys[3]; // Assuming a 3-node B-tree for simplicity
    int children[4];
    int count;
};

// Global memory for B-tree nodes
__device__ BTreeNode btree[MAX_NODES];

// Kernel to insert a key into the B-tree
__global__ void insertKeyKernel(int key) {
    int tid = threadIdx.x;
    if (tid >= MAX_NODES) return;

    // Use atomics to safely update the B-tree
    int targetNode = atomicCAS(&btree[0].children[0], -1, tid);
    if (targetNode != -1) {
        // If the root is not empty, find the correct node to insert
        while (btree[targetNode].count == 3) {
            int i = 0;
            while (i < btree[targetNode].count && key > btree[targetNode].keys[i]) i++;
            targetNode = atomicCAS(&btree[targetNode].children[i], -1, tid);
        }
    }

    // Insert the key into the found node
    int i = 0;
    while (i < btree[targetNode].count && key > btree[targetNode].keys[i]) i++;
    for (int j = btree[targetNode].count; j > i; j--) {
        btree[targetNode].keys[j] = btree[targetNode].keys[j - 1];
    }
    btree[targetNode].keys[i] = key;
    atomicAdd(&btree[targetNode].count, 1);
}

// Kernel to print the B-tree nodes
__global__ void printBTreeKernel() {
    int tid = threadIdx.x;
    if (tid >= MAX_NODES) return;

    printf("Node %d: ", tid);
    for (int i = 0; i < btree[tid].count; i++) {
        printf("%d ", btree[tid].keys[i]);
    }
    printf("\n");
}

int main() {
    // Initialize the B-tree in device memory
    hipError_t err;
    BTreeNode h_btree[MAX_NODES];
    for (int i = 0; i < MAX_NODES; i++) {
        h_btree[i].count = 0;
        for (int j = 0; j < 4; j++) h_btree[i].children[j] = -1;
    }
    err = hipMemcpy(btree, h_btree, sizeof(h_btree), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy B-tree to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch the kernel to insert keys into the B-tree
    int keys[] = {10, 20, 5, 6, 12, 30, 7, 17};
    for (int key : keys) {
        insertKeyKernel<<<1, MAX_NODES>>>(key);
        hipDeviceSynchronize(); // Ensure all keys are inserted before printing
        if (err != hipSuccess) {
            std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Launch the kernel to print the B-tree
    printBTreeKernel<<<1, MAX_NODES>>>();
    hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free resources and exit
    hipDeviceReset();
    return 0;
}