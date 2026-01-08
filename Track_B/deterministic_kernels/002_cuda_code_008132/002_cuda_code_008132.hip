#include <iostream>
#include <hip/hip_runtime.h>

#define TABLE_SIZE 64
#define BLOCK_SIZE 32

// Kernel to insert elements into the hash table
__global__ void insertIntoHashTable(int* table, int* keys, int* values, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        int key = keys[idx];
        int value = values[idx];
        int hash = key % TABLE_SIZE; // Simple hash function

        // Linear probing to resolve collisions
        while (atomicCAS(&table[hash], 0, key + 1) != 0) {
            hash = (hash + 1) % TABLE_SIZE;
        }
        table[hash + TABLE_SIZE] = value; // Store value in the second half of the table
    }
}

// Kernel to lookup values in the hash table
__global__ void lookupHashTable(int* table, int* keys, int* results, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        int key = keys[idx];
        int hash = key % TABLE_SIZE; // Simple hash function

        // Linear probing to find the key
        while (table[hash] != key + 1 && table[hash] != 0) {
            hash = (hash + 1) % TABLE_SIZE;
        }
        results[idx] = (table[hash] == key + 1) ? table[hash + TABLE_SIZE] : -1; // Return value or -1 if not found
    }
}

int main() {
    int numElements = 32; // Number of elements to insert and lookup
    int* h_keys = new int[numElements];
    int* h_values = new int[numElements];
    int* h_results = new int[numElements];

    // Initialize keys and values
    for (int i = 0; i < numElements; ++i) {
        h_keys[i] = i;
        h_values[i] = i * 10;
        h_results[i] = -1;
    }

    int* d_table;
    int* d_keys;
    int* d_values;
    int* d_results;

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_table, sizeof(int) * TABLE_SIZE * 2);
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for table: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_keys, sizeof(int) * numElements);
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_values, sizeof(int) * numElements);
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for values: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_results, sizeof(int) * numElements);
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for results: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy data to device
    err = hipMemcpy(d_keys, h_keys, sizeof(int) * numElements, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying keys to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMemcpy(d_values, h_values, sizeof(int) * numElements, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying values to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize table to 0
    err = hipMemset(d_table, 0, sizeof(int) * TABLE_SIZE * 2);
    if (err != hipSuccess) {
        std::cerr << "Error initializing table: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch insert kernel
    int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    insertIntoHashTable<<<numBlocks, BLOCK_SIZE>>>(d_table, d_keys, d_values, numElements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching insert kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch lookup kernel
    lookupHashTable<<<numBlocks, BLOCK_SIZE>>>(d_table, d_keys, d_results, numElements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching lookup kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_results, d_results, sizeof(int) * numElements, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying results to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Print results
    std::cout << "Lookup results:" << std::endl;
    for (int i = 0; i < numElements; ++i) {
        std::cout << "Key: " << h_keys[i] << ", Value: " << h_results[i] << std::endl;
    }

    // Free device memory
    hipFree(d_table);
    hipFree(d_keys);
    hipFree(d_values);
    hipFree(d_results);

    // Free host memory
    delete[] h_keys;
    delete[] h_values;
    delete[] h_results;

    return 0;
}