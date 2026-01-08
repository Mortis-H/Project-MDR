#include <iostream>
#include <hip/hip_runtime.h>


#define BUCKET_SIZE 256
#define BUCKET_COUNT 8192
#define TABLE_SIZE (BUCKET_COUNT * BUCKET_SIZE)

__device__ unsigned int hash1(unsigned int key) {
    return key % BUCKET_COUNT;
}

__device__ unsigned int hash2(unsigned int key) {
    return 1 + (key % (BUCKET_COUNT - 1));
}

__global__ void insertKernel(unsigned int *table, unsigned int *keys, unsigned int numKeys) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numKeys) return;

    unsigned int key = keys[idx];
    unsigned int h1 = hash1(key);
    unsigned int h2 = hash2(key);
    unsigned int i = 0;

    while (true) {
        unsigned int pos = (h1 + i * h2) % BUCKET_COUNT;
        unsigned int bucketPos = pos * BUCKET_SIZE;
        for (unsigned int j = 0; j < BUCKET_SIZE; ++j) {
            unsigned int slot = bucketPos + j;
            unsigned int old = atomicCAS(&table[slot], 0, key);
            if (old == 0) return; // Successfully inserted
            if (old == key) return; // Key already exists
        }
        i++;
    }
}

__global__ void searchKernel(unsigned int *table, unsigned int *keys, bool *results, unsigned int numKeys) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numKeys) return;

    unsigned int key = keys[idx];
    unsigned int h1 = hash1(key);
    unsigned int h2 = hash2(key);
    unsigned int i = 0;

    while (true) {
        unsigned int pos = (h1 + i * h2) % BUCKET_COUNT;
        unsigned int bucketPos = pos * BUCKET_SIZE;
        for (unsigned int j = 0; j < BUCKET_SIZE; ++j) {
            unsigned int slot = bucketPos + j;
            if (table[slot] == key) {
                results[idx] = true;
                return;
            }
            if (table[slot] == 0) {
                results[idx] = false;
                return;
            }
        }
        i++;
    }
}

int main() {
    unsigned int numKeys = 1024;
    unsigned int *h_keys = new unsigned int[numKeys];
    bool *h_results = new bool[numKeys];
    for (unsigned int i = 0; i < numKeys; ++i) {
        h_keys[i] = i * 3 + 1; // Example keys
        h_results[i] = false;
    }

    unsigned int *d_keys, *d_table;
    bool *d_results;
    hipMalloc((void**)&d_keys, numKeys * sizeof(unsigned int));
    hipMalloc((void**)&d_results, numKeys * sizeof(bool));
    hipMalloc((void**)&d_table, TABLE_SIZE * sizeof(unsigned int));
    hipMemset(d_table, 0, TABLE_SIZE * sizeof(unsigned int));

    hipMemcpy(d_keys, h_keys, numKeys * sizeof(unsigned int), hipMemcpyHostToDevice);

    unsigned int blockSize = 256;
    unsigned int numBlocks = (numKeys + blockSize - 1) / blockSize;

    // Insert keys into hash table
    insertKernel<<<numBlocks, blockSize>>>(d_table, d_keys, numKeys);
    hipDeviceSynchronize();

    // Search for keys in hash table
    searchKernel<<<numBlocks, blockSize>>>(d_table, d_keys, d_results, numKeys);
    hipDeviceSynchronize();

    hipMemcpy(h_results, d_results, numKeys * sizeof(bool), hipMemcpyDeviceToHost);

    // Verify results
    bool allFound = true;
    for (unsigned int i = 0; i < numKeys; ++i) {
        if (!h_results[i]) {
            allFound = false;
            break;
        }
    }

    if (allFound) {
        std::cout << "All keys found successfully." << std::endl;
    } else {
        std::cout << "Some keys were not found." << std::endl;
    }

    hipFree(d_keys);
    hipFree(d_results);
    hipFree(d_table);
    delete[] h_keys;
    delete[] h_results;

    return 0;
}