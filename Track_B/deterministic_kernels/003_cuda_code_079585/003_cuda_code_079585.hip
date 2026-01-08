#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define the maximum level for the skip list
#define MAX_LEVEL 16

// Define the number of elements in the skip list
#define NUM_ELEMENTS 256

// Define the warp size
#define WARP_SIZE 32

// Node structure for the skip list
struct Node {
    int value;
    int level;
    Node* forward[MAX_LEVEL];
};

// Shared memory for storing nodes
__shared__ Node sharedNodes[NUM_ELEMENTS];

// Kernel to initialize nodes in shared memory
__global__ void initializeNodes(Node* d_nodes) {
    int idx = threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        d_nodes[idx].value = idx; // Initialize with index value
        d_nodes[idx].level = 0;   // Initialize level to 0
        for (int i = 0; i < MAX_LEVEL; ++i) {
            d_nodes[idx].forward[i] = nullptr;
        }
    }
}

// Kernel to build the skip list
__global__ void buildSkipList(Node* d_nodes) {
    extern __shared__ Node sharedNodes[];
    int idx = threadIdx.x;

    // Load nodes into shared memory
    if (idx < NUM_ELEMENTS) {
        sharedNodes[idx] = d_nodes[idx];
    }
    __syncthreads();

    // Each warp builds a part of the skip list
    int warp_id = idx / WARP_SIZE;
    int lane_id = idx % WARP_SIZE;

    // Simple random level generation (for demonstration purposes)
    hiprandState state;
    hiprand_init(clock64(), idx, 0, &state);
    int randomLevel = hiprand(&state) % MAX_LEVEL;

    if (idx < NUM_ELEMENTS) {
        sharedNodes[idx].level = randomLevel;
        for (int level = 0; level <= randomLevel; ++level) {
            // Simple insertion logic (for demonstration purposes)
            int nextIdx = (idx + 1) % NUM_ELEMENTS;
            sharedNodes[idx].forward[level] = &sharedNodes[nextIdx];
        }
    }
    __syncthreads();

    // Copy back to global memory
    if (idx < NUM_ELEMENTS) {
        d_nodes[idx] = sharedNodes[idx];
    }
}

// Kernel to search for a value in the skip list
__global__ void searchSkipList(Node* d_nodes, int searchValue, int* result) {
    extern __shared__ Node sharedNodes[];
    int idx = threadIdx.x;

    // Load nodes into shared memory
    if (idx < NUM_ELEMENTS) {
        sharedNodes[idx] = d_nodes[idx];
    }
    __syncthreads();

    // Each warp performs a search
    int warp_id = idx / WARP_SIZE;
    int lane_id = idx % WARP_SIZE;

    if (lane_id == 0) {
        Node* current = &sharedNodes[0];
        for (int level = MAX_LEVEL - 1; level >= 0; --level) {
            while (current->forward[level] != nullptr && current->forward[level]->value < searchValue) {
                current = current->forward[level];
            }
        }
        current = current->forward[0];
        if (current != nullptr && current->value == searchValue) {
            *result = current->value;
        } else {
            *result = -1; // Not found
        }
    }
}

int main() {
    Node* d_nodes;
    int* d_result;
    int h_result = -1;

    // Allocate memory on the device
    hipError_t err = hipMalloc(&d_nodes, NUM_ELEMENTS * sizeof(Node));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for nodes" << std::endl;
        return -1;
    }

    err = hipMalloc(&d_result, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for result" << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Initialize nodes on the device
    initializeNodes<<<1, NUM_ELEMENTS>>>(d_nodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_result);
        return -1;
    }

    // Build the skip list on the device
    buildSkipList<<<1, NUM_ELEMENTS, NUM_ELEMENTS * sizeof(Node)>>>(d_nodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_result);
        return -1;
    }

    // Search for a value in the skip list
    int searchValue = 128; // Example search value
    searchSkipList<<<1, 1, NUM_ELEMENTS * sizeof(Node)>>>(d_nodes, searchValue, d_result);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_result);
        return -1;
    }

    // Copy result back to host
    hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost);
    if (h_result != -1) {
        std::cout << "Value " << h_result << " found in the skip list." << std::endl;
    } else {
        std::cout << "Value not found in the skip list." << std::endl;
    }

    // Free device memory
    hipFree(d_nodes);
    hipFree(d_result);

    return 0;
}