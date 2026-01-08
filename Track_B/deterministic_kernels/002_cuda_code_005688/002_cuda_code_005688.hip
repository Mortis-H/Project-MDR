#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <vector>

namespace cg = cooperative_groups;

#define BUCKETS 256
#define MAX_ELEMENTS 1024
#define ELEMENTS_PER_THREAD 8

// Node structure for separate chaining
struct Node {
    int key;
    int value;
    int next;
};

// Global memory for hash table
__device__ Node d_nodes[MAX_ELEMENTS];
__device__ int d_heads[BUCKETS];

// Kernel to insert elements into the hash table
__global__ void insertKernel(int* keys, int* values, int numElements) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < numElements; i += stride) {
        int key = keys[i];
        int value = values[i];
        int bucket = key % BUCKETS;

        // Use atomic to find the next available node
        int nodeIndex = atomicAdd(&d_heads[bucket], 1);

        if (nodeIndex < MAX_ELEMENTS) {
            d_nodes[nodeIndex].key = key;
            d_nodes[nodeIndex].value = value;
            d_nodes[nodeIndex].next = -1; // Initialize next pointer to -1

            // Insert node at the beginning of the list
            int oldHead = atomicExch(&d_heads[bucket], nodeIndex);
            d_nodes[nodeIndex].next = oldHead;
        }
    }
}

// Kernel to search for a key in the hash table
__global__ void searchKernel(int* keys, int* results, int numElements) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < numElements; i += stride) {
        int key = keys[i];
        int bucket = key % BUCKETS;
        int nodeIndex = d_heads[bucket];

        while (nodeIndex != -1) {
            if (d_nodes[nodeIndex].key == key) {
                results[i] = d_nodes[nodeIndex].value;
                break;
            }
            nodeIndex = d_nodes[nodeIndex].next;
        }

        if (nodeIndex == -1) {
            results[i] = -1; // Key not found
        }
    }
}

int main() {
    int numElements = ELEMENTS_PER_THREAD * 256; // Total number of elements to insert

    // Host memory for keys and values
    std::vector<int> h_keys(numElements);
    std::vector<int> h_values(numElements);
    std::vector<int> h_results(numElements);

    // Initialize keys and values
    for (int i = 0; i < numElements; ++i) {
        h_keys[i] = i; // Simple key generation
        h_values[i] = i * 10; // Simple value generation
    }

    // Device memory for keys and values
    int* d_keys;
    int* d_values;
    int* d_results;

    // Allocate device memory
    hipError_t err = hipMalloc(&d_keys, numElements * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for keys" << std::endl;
        return -1;
    }

    err = hipMalloc(&d_values, numElements * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for values" << std::endl;
        hipFree(d_keys);
        return -1;
    }

    err = hipMalloc(&d_results, numElements * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for results" << std::endl;
        hipFree(d_keys);
        hipFree(d_values);
        return -1;
    }

    // Copy keys and values to device
    err = hipMemcpy(d_keys, h_keys.data(), numElements * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy keys to device" << std::endl;
        hipFree(d_keys);
        hipFree(d_values);
        hipFree(d_results);
        return -1;
    }

    err = hipMemcpy(d_values, h_values.data(), numElements * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy values to device" << std::endl;
        hipFree(d_keys);
        hipFree(d_values);
        hipFree(d_results);
        return -1;
    }

    // Initialize device memory for hash table
    err = hipMemset(d_heads, -1, BUCKETS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize hash table heads" << std::endl;
        hipFree(d_keys);
        hipFree(d_values);
        hipFree(d_results);
        return -1;
    }

    // Launch insert kernel
    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;

    insertKernel<<<numBlocks, blockSize>>>(d_keys, d_values, numElements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch insert kernel" << std::endl;
        hipFree(d_keys);
        hipFree(d_values);
        hipFree(d_results);
        return -1;
    }

    // Launch search kernel
    searchKernel<<<numBlocks, blockSize>>>(d_keys, d_results, numElements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch search kernel" << std::endl;
        hipFree(d_keys);
        hipFree(d_values);
        hipFree(d_results);
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_results.data(), d_results, numElements * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy results from device" << std::endl;
        hipFree(d_keys);
        hipFree(d_values);
        hipFree(d_results);
        return -1;
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_values);
    hipFree(d_results);

    // Verify results
    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        if (h_results[i] != h_values[i]) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << h_values[i] << ", got " << h_results[i] << std::endl;
            break;
        }
    }

    if (success) {
        std::cout << "All keys found successfully!" << std::endl;
    }

    return 0;
}