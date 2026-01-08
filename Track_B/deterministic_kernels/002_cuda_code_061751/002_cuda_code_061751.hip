#include <iostream>
#include <hip/hip_runtime.h>

#define BUCKET_COUNT 32
#define TABLE_SIZE (BUCKET_COUNT * 2) // Double the bucket count for open addressing

__constant__ int d_bucket_count = BUCKET_COUNT;

// Kernel to insert elements into the hash table using linear probing
__global__ void insertKernel(int *hashTable, int *keys, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys) {
        int key = keys[idx];
        int hash = key % d_bucket_count;
        int probe = 0;
        while (probe < TABLE_SIZE) {
            int pos = (hash + probe) % TABLE_SIZE;
            int expected = -1;
            if (atomicCAS(&hashTable[pos], expected, key) == expected) {
                break;
            }
            probe++;
        }
    }
}

// Kernel to search for elements in the hash table using linear probing
__global__ void searchKernel(int *hashTable, int *keys, int *results, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys) {
        int key = keys[idx];
        int hash = key % d_bucket_count;
        int probe = 0;
        while (probe < TABLE_SIZE) {
            int pos = (hash + probe) % TABLE_SIZE;
            if (hashTable[pos] == key) {
                results[idx] = 1; // Key found
                break;
            } else if (hashTable[pos] == -1) {
                results[idx] = 0; // Key not found
                break;
            }
            probe++;
        }
    }
}

int main() {
    int numKeys = 16;
    int *h_keys = new int[numKeys];
    int *h_results = new int[numKeys];
    int *d_hashTable, *d_keys, *d_results;

    // Initialize keys
    for (int i = 0; i < numKeys; i++) {
        h_keys[i] = i * 3 + 1; // Arbitrary keys
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_hashTable, TABLE_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate hash table: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_keys, numKeys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_results, numKeys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate results: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize hash table to -1 (empty)
    err = hipMemset(d_hashTable, -1, TABLE_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize hash table: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy keys to device
    err = hipMemcpy(d_keys, h_keys, numKeys * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy keys to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Insert keys into hash table
    int blockSize = 256;
    int numBlocks = (numKeys + blockSize - 1) / blockSize;
    insertKernel<<<numBlocks, blockSize>>>(d_hashTable, d_keys, numKeys);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Search for keys in hash table
    searchKernel<<<numBlocks, blockSize>>>(d_hashTable, d_keys, d_results, numKeys);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_results, d_results, numKeys * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy results to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Print results
    for (int i = 0; i < numKeys; i++) {
        std::cout << "Key " << h_keys[i] << " found: " << (h_results[i] ? "Yes" : "No") << std::endl;
    }

    // Free device memory
    hipFree(d_hashTable);
    hipFree(d_keys);
    hipFree(d_results);

    // Free host memory
    delete[] h_keys;
    delete[] h_results;

    return 0;
}