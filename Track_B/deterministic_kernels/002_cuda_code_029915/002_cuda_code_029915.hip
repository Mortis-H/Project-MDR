#include <iostream>
#include <hip/hip_runtime.h>


// Define the maximum level of the skip list
#define MAX_LEVEL 16

// Node structure for the skip list
struct Node {
    int value;
    Node* forward[MAX_LEVEL];
};

// Kernel to initialize the skip list
__global__ void initSkipList(Node* nodes, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        nodes[idx].value = idx; // Initialize with index as value for simplicity
        for (int i = 0; i < MAX_LEVEL; ++i) {
            nodes[idx].forward[i] = nullptr;
        }
    }
}

// Kernel to perform a simple search in the skip list
__global__ void searchSkipList(Node* nodes, int numNodes, int target, bool* found) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) { // Only one thread performs the search
        Node* current = &nodes[0];
        for (int i = MAX_LEVEL - 1; i >= 0; --i) {
            while (current->forward[i] != nullptr && current->forward[i]->value < target) {
                current = current->forward[i];
            }
        }
        *found = (current->forward[0] != nullptr && current->forward[0]->value == target);
    }
}

int main() {
    const int numNodes = 512;
    Node* d_nodes;
    bool* d_found;
    bool h_found;

    // Allocate memory on the device
    hipError_t err = hipMalloc(&d_nodes, numNodes * sizeof(Node));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for nodes: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc(&d_found, sizeof(bool));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for found flag: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Initialize the skip list on the device
    int threadsPerBlock = 256;
    int blocksPerGrid = (numNodes + threadsPerBlock - 1) / threadsPerBlock;
    initSkipList<<<blocksPerGrid, threadsPerBlock>>>(d_nodes, numNodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching initSkipList kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_found);
        return -1;
    }

    // Perform a search on the device
    int target = 256; // Example target value
    searchSkipList<<<1, 1>>>(d_nodes, numNodes, target, d_found);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching searchSkipList kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_found);
        return -1;
    }

    // Copy the result back to the host
    err = hipMemcpy(&h_found, d_found, sizeof(bool), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying found flag to host: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_found);
        return -1;
    }

    // Output the result
    std::cout << "Target " << target << (h_found ? " found" : " not found") << " in the skip list." << std::endl;

    // Free device memory
    hipFree(d_nodes);
    hipFree(d_found);

    return 0;
}