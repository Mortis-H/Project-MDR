#include <iostream>
#include <hip/hip_runtime.h>

#define NUM_BUCKETS 16384
#define BLOCK_SIZE 256
#define HASH_TABLE_SIZE (NUM_BUCKETS * 2) // Each bucket has a key and a value

__global__ void insertKernel(int *hashTable, int *keys, int *values, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    int key = keys[idx];
    int value = values[idx];
    int hash = key % NUM_BUCKETS;
    int step = 1;

    // Quadratic probing
    while (true) {
        int pos = (hash + step * step) % NUM_BUCKETS;
        int posKey = pos * 2;
        int posValue = posKey + 1;

        int oldKey = atomicCAS(&hashTable[posKey], 0, key);
        if (oldKey == 0) {
            hashTable[posValue] = value;
            break;
        } else if (oldKey == key) {
            hashTable[posValue] = value; // Update value if key already exists
            break;
        }
        step++;
    }
}

__global__ void searchKernel(int *hashTable, int *searchKeys, int *results, int numSearches) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSearches) return;

    int key = searchKeys[idx];
    int hash = key % NUM_BUCKETS;
    int step = 1;

    // Quadratic probing
    while (true) {
        int pos = (hash + step * step) % NUM_BUCKETS;
        int posKey = pos * 2;

        int currentKey = hashTable[posKey];
        if (currentKey == key) {
            results[idx] = hashTable[posKey + 1]; // Found the key, return the value
            break;
        } else if (currentKey == 0) {
            results[idx] = -1; // Key not found
            break;
        }
        step++;
    }
}

int main() {
    int numElements = 8192;
    int numSearches = 1024;

    // Allocate host memory
    int *h_keys = new int[numElements];
    int *h_values = new int[numElements];
    int *h_searchKeys = new int[numSearches];
    int *h_results = new int[numSearches];

    // Initialize host memory
    for (int i = 0; i < numElements; i++) {
        h_keys[i] = i;
        h_values[i] = i * 2;
    }
    for (int i = 0; i < numSearches; i++) {
        h_searchKeys[i] = i;
    }

    // Allocate device memory
    int *d_hashTable, *d_keys, *d_values, *d_searchKeys, *d_results;
    hipMalloc(&d_hashTable, HASH_TABLE_SIZE * sizeof(int));
    hipMalloc(&d_keys, numElements * sizeof(int));
    hipMalloc(&d_values, numElements * sizeof(int));
    hipMalloc(&d_searchKeys, numSearches * sizeof(int));
    hipMalloc(&d_results, numSearches * sizeof(int));

    // Initialize hash table to 0
    hipMemset(d_hashTable, 0, HASH_TABLE_SIZE * sizeof(int));

    // Asynchronous memory copy to device
    hipMemcpyAsync(d_keys, h_keys, numElements * sizeof(int), hipMemcpyHostToDevice, 0);
    hipMemcpyAsync(d_values, h_values, numElements * sizeof(int), hipMemcpyHostToDevice, 0);
    hipMemcpyAsync(d_searchKeys, h_searchKeys, numSearches * sizeof(int), hipMemcpyHostToDevice, 0);

    // Launch insert kernel
    int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    insertKernel<<<numBlocks, BLOCK_SIZE>>>(d_hashTable, d_keys, d_values, numElements);
    hipDeviceSynchronize();

    // Launch search kernel
    numBlocks = (numSearches + BLOCK_SIZE - 1) / BLOCK_SIZE;
    searchKernel<<<numBlocks, BLOCK_SIZE>>>(d_hashTable, d_searchKeys, d_results, numSearches);
    hipDeviceSynchronize();

    // Asynchronous memory copy back to host
    hipMemcpyAsync(h_results, d_results, numSearches * sizeof(int), hipMemcpyDeviceToHost, 0);
    hipDeviceSynchronize();

    // Verify results
    bool success = true;
    for (int i = 0; i < numSearches; i++) {
        if (h_results[i] != h_searchKeys[i] * 2) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "All searches were successful!" << std::endl;
    } else {
        std::cout << "There were errors in the search results." << std::endl;
    }

    // Free device memory
    hipFree(d_hashTable);
    hipFree(d_keys);
    hipFree(d_values);
    hipFree(d_searchKeys);
    hipFree(d_results);

    // Free host memory
    delete[] h_keys;
    delete[] h_values;
    delete[] h_searchKeys;
    delete[] h_results;

    return 0;
}