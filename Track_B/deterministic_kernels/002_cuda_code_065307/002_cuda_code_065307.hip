// hash_table.cu
// A basic CUDA implementation of a hash table with 1024 elements
// Target Architecture: Volta

#include <iostream>
#include <hip/hip_runtime.h>

#define TABLE_SIZE 1024

// Hash function to compute the index for a given key
__device__ unsigned int hash(unsigned int key) {
    return key % TABLE_SIZE;
}

// Kernel to insert keys into the hash table
__global__ void insertIntoHashTable(unsigned int* keys, unsigned int* hashTable, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys) {
        unsigned int key = keys[idx];
        unsigned int index = hash(key);
        // Simple linear probing for collision resolution
        while (atomicCAS(&hashTable[index], 0, key) != 0) {
            index = (index + 1) % TABLE_SIZE;
        }
    }
}

// Kernel to search for keys in the hash table
__global__ void searchHashTable(unsigned int* keys, unsigned int* hashTable, int numKeys, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys) {
        unsigned int key = keys[idx];
        unsigned int index = hash(key);
        // Simple linear probing to find the key
        while (hashTable[index] != key && hashTable[index] != 0) {
            index = (index + 1) % TABLE_SIZE;
        }
        results[idx] = (hashTable[index] == key) ? 1 : 0;
    }
}

int main() {
    // Host data
    unsigned int* h_keys;
    unsigned int* h_hashTable;
    int* h_results;

    // Device data
    unsigned int* d_keys;
    unsigned int* d_hashTable;
    int* d_results;

    // Number of keys to insert and search
    int numKeys = 256;

    // Allocate host memory
    h_keys = new unsigned int[numKeys];
    h_hashTable = new unsigned int[TABLE_SIZE];
    h_results = new int[numKeys];

    // Initialize hash table on host
    for (int i = 0; i < TABLE_SIZE; i++) {
        h_hashTable[i] = 0;
    }

    // Initialize keys on host
    for (int i = 0; i < numKeys; i++) {
        h_keys[i] = i * 3; // Example keys
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_keys, numKeys * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for keys" << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_hashTable, TABLE_SIZE * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for hash table" << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_results, numKeys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for results" << std::endl;
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_keys, h_keys, numKeys * sizeof(unsigned int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy keys to device" << std::endl;
        return -1;
    }

    err = hipMemcpy(d_hashTable, h_hashTable, TABLE_SIZE * sizeof(unsigned int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy hash table to device" << std::endl;
        return -1;
    }

    // Define grid and block dimensions
    dim3 blockSize(256);
    dim3 gridSize((numKeys + blockSize.x - 1) / blockSize.x);

    // Launch kernel to insert keys into hash table
    insertIntoHashTable<<<gridSize, blockSize>>>(d_keys, d_hashTable, numKeys);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch insert kernel" << std::endl;
        return -1;
    }

    // Launch kernel to search for keys in hash table
    searchHashTable<<<gridSize, blockSize>>>(d_keys, d_hashTable, numKeys, d_results);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch search kernel" << std::endl;
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_results, d_results, numKeys * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy results from device" << std::endl;
        return -1;
    }

    // Print results
    std::cout << "Search results:" << std::endl;
    for (int i = 0; i < numKeys; i++) {
        std::cout << "Key " << h_keys[i] << ": " << (h_results[i] ? "Found" : "Not Found") << std::endl;
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_hashTable);
    hipFree(d_results);

    // Free host memory
    delete[] h_keys;
    delete[] h_hashTable;
    delete[] h_results;

    return 0;
}