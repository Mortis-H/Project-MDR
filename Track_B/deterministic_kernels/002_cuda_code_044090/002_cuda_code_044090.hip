#include <iostream>
#include <hip/hip_runtime.h>

#define BUCKET_SIZE 256
#define TABLE_SIZE (256 * BUCKET_SIZE)

// Hash functions for double hashing
__device__ unsigned int hash1(unsigned int key) {
    return key % TABLE_SIZE;
}

__device__ unsigned int hash2(unsigned int key) {
    return 1 + (key % (TABLE_SIZE - 1));
}

// Kernel to insert keys into the hash table using double hashing
__global__ void insertKeys(unsigned int* keys, unsigned int* table, unsigned int numKeys) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numKeys) return;

    unsigned int key = keys[idx];
    unsigned int h1 = hash1(key);
    unsigned int h2 = hash2(key);
    unsigned int i = 0;
    unsigned int pos;

    // Double hashing to find an empty slot
    do {
        pos = (h1 + i * h2) % TABLE_SIZE;
        i++;
    } while (atomicCAS(&table[pos], 0, key) != 0);
}

// Kernel to search for keys in the hash table using double hashing
__global__ void searchKeys(unsigned int* keys, unsigned int* table, unsigned int numKeys, unsigned int* results) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numKeys) return;

    unsigned int key = keys[idx];
    unsigned int h1 = hash1(key);
    unsigned int h2 = hash2(key);
    unsigned int i = 0;
    unsigned int pos;

    // Double hashing to find the key
    do {
        pos = (h1 + i * h2) % TABLE_SIZE;
        if (table[pos] == key) {
            results[idx] = 1; // Key found
            return;
        }
        i++;
    } while (table[pos] != 0);
    results[idx] = 0; // Key not found
}

int main() {
    unsigned int numKeys = 1024;
    unsigned int* keys;
    unsigned int* d_keys;
    unsigned int* d_table;
    unsigned int* d_results;
    unsigned int* results = new unsigned int[numKeys];

    // Allocate host memory
    keys = new unsigned int[numKeys];
    for (unsigned int i = 0; i < numKeys; i++) {
        keys[i] = i * 3 + 1; // Generate some keys
    }

    // Allocate device memory
    hipError_t err = hipMalloc(&d_keys, numKeys * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for keys" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_table, TABLE_SIZE * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for table" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_results, numKeys * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for results" << std::endl;
        return -1;
    }

    // Initialize table to 0
    err = hipMemset(d_table, 0, TABLE_SIZE * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize table" << std::endl;
        return -1;
    }

    // Copy keys to device
    err = hipMemcpy(d_keys, keys, numKeys * sizeof(unsigned int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy keys to device" << std::endl;
        return -1;
    }

    // Set up execution configuration
    unsigned int blockSize = 256;
    unsigned int numBlocks = (numKeys + blockSize - 1) / blockSize;

    // Insert keys into hash table
    insertKeys<<<numBlocks, blockSize>>>(d_keys, d_table, numKeys);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch insertKeys kernel" << std::endl;
        return -1;
    }

    // Search for keys in hash table
    searchKeys<<<numBlocks, blockSize>>>(d_keys, d_table, numKeys, d_results);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch searchKeys kernel" << std::endl;
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(results, d_results, numKeys * sizeof(unsigned int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy results to host" << std::endl;
        return -1;
    }

    // Verify results
    bool success = true;
    for (unsigned int i = 0; i < numKeys; i++) {
        if (results[i] != 1) {
            std::cerr << "Key " << keys[i] << " not found" << std::endl;
            success = false;
        }
    }
    if (success) {
        std::cout << "All keys found successfully" << std::endl;
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_table);
    hipFree(d_results);

    // Free host memory
    delete[] keys;
    delete[] results;

    return 0;
}