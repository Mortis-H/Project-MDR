#include <iostream>
#include <hip/hip_runtime.h>

#define BUCKET_SIZE 16
#define NUM_BUCKETS 4096
#define TABLE_SIZE (NUM_BUCKETS * BUCKET_SIZE)

__device__ unsigned int hash1(unsigned int key) {
    key = ~key + (key << 15);
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057;
    key = key ^ (key >> 16);
    return key % NUM_BUCKETS;
}

__device__ unsigned int hash2(unsigned int key) {
    key = key ^ (key >> 3);
    key = key * (key + 3);
    key = key ^ (key >> 4);
    key = key * (key + 5);
    key = key ^ (key >> 15);
    return (key % (NUM_BUCKETS - 1)) + 1; // Ensure hash2 is not zero
}

__global__ void insertKernel(unsigned int* table, unsigned int* keys, int numKeys) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numKeys) return;

    unsigned int key = keys[tid];
    unsigned int h1 = hash1(key);
    unsigned int h2 = hash2(key);
    unsigned int index = h1 * BUCKET_SIZE;

    // Use warp-level primitives for conflict resolution
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        unsigned int slot = index + i;
        unsigned int currentKey = atomicCAS(&table[slot], 0, key);
        if (currentKey == 0 || currentKey == key) {
            break;
        } else {
            h1 = (h1 + h2) % NUM_BUCKETS;
            index = h1 * BUCKET_SIZE;
        }
    }
}

__global__ void searchKernel(unsigned int* table, unsigned int* keys, int numKeys, int* results) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numKeys) return;

    unsigned int key = keys[tid];
    unsigned int h1 = hash1(key);
    unsigned int h2 = hash2(key);
    unsigned int index = h1 * BUCKET_SIZE;

    // Use warp-level primitives for conflict resolution
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        unsigned int slot = index + i;
        if (table[slot] == key) {
            results[tid] = 1;
            return;
        } else if (table[slot] == 0) {
            results[tid] = 0;
            return;
        } else {
            h1 = (h1 + h2) % NUM_BUCKETS;
            index = h1 * BUCKET_SIZE;
        }
    }
    results[tid] = 0;
}

int main() {
    const int numKeys = 1024;
    unsigned int* h_keys = new unsigned int[numKeys];
    int* h_results = new int[numKeys];
    unsigned int* d_keys;
    unsigned int* d_table;
    int* d_results;

    // Initialize keys
    for (int i = 0; i < numKeys; ++i) {
        h_keys[i] = i * 13; // Simple key generation
    }

    // Allocate device memory
    hipError_t err = hipMalloc(&d_table, TABLE_SIZE * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for table: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&d_keys, numKeys * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&d_results, numKeys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for results: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize table to zero
    err = hipMemset(d_table, 0, TABLE_SIZE * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Error setting table to zero: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy keys to device
    err = hipMemcpy(d_keys, h_keys, numKeys * sizeof(unsigned int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying keys to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Insert keys into hash table
    int blockSize = 256;
    int numBlocks = (numKeys + blockSize - 1) / blockSize;
    insertKernel<<<numBlocks, blockSize>>>(d_table, d_keys, numKeys);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error in insertKernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Search for keys in hash table
    searchKernel<<<numBlocks, blockSize>>>(d_table, d_keys, numKeys, d_results);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error in searchKernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_results, d_results, numKeys * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying results to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Verify results
    bool allFound = true;
    for (int i = 0; i < numKeys; ++i) {
        if (!h_results[i]) {
            allFound = false;
            break;
        }
    }

    if (allFound) {
        std::cout << "All keys found successfully in the hash table." << std::endl;
    } else {
        std::cout << "Some keys were not found in the hash table." << std::endl;
    }

    // Free device memory
    hipFree(d_table);
    hipFree(d_keys);
    hipFree(d_results);

    // Free host memory
    delete[] h_keys;
    delete[] h_results;

    return 0;
}