#include <iostream>
#include <hip/hip_runtime.h>

#define NUM_BUCKETS 1024
#define MAX_KEY 10000

// Kernel to insert keys into the hash table using linear probing
__global__ void insertKeys(int *keys, int *hashTable, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys) {
        int key = keys[idx];
        int hash = key % NUM_BUCKETS;
        int i = 0;

        // Linear probing to find an empty slot
        while (hashTable[hash] != -1) {
            hash = (hash + 1) % NUM_BUCKETS;
            i++;
            // Prevent infinite loop in case of full table
            if (i == NUM_BUCKETS) break;
        }

        // Insert key if a slot is found
        if (hashTable[hash] == -1) {
            hashTable[hash] = key;
        }
    }
}

// Kernel to search for keys in the hash table using linear probing
__global__ void searchKeys(int *keys, int *hashTable, int *results, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys) {
        int key = keys[idx];
        int hash = key % NUM_BUCKETS;
        int i = 0;

        // Linear probing to find the key
        while (hashTable[hash] != key && hashTable[hash] != -1) {
            hash = (hash + 1) % NUM_BUCKETS;
            i++;
            // Prevent infinite loop in case of full table
            if (i == NUM_BUCKETS) break;
        }

        // Store result: 1 if found, 0 if not found
        results[idx] = (hashTable[hash] == key) ? 1 : 0;
    }
}

int main() {
    int numKeys = 512; // Number of keys to insert and search
    int *h_keys, *h_hashTable, *h_results;
    int *d_keys, *d_hashTable, *d_results;

    // Allocate host memory
    h_keys = (int *)malloc(numKeys * sizeof(int));
    h_hashTable = (int *)malloc(NUM_BUCKETS * sizeof(int));
    h_results = (int *)malloc(numKeys * sizeof(int));

    // Initialize hash table with -1 (empty)
    for (int i = 0; i < NUM_BUCKETS; i++) {
        h_hashTable[i] = -1;
    }

    // Initialize keys with random values
    for (int i = 0; i < numKeys; i++) {
        h_keys[i] = rand() % MAX_KEY;
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void **)&d_keys, numKeys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for keys" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_hashTable, NUM_BUCKETS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for hash table" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_results, numKeys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for results" << std::endl;
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_keys, h_keys, numKeys * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy keys to device" << std::endl;
        return -1;
    }
    err = hipMemcpy(d_hashTable, h_hashTable, NUM_BUCKETS * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy hash table to device" << std::endl;
        return -1;
    }

    // Launch kernel to insert keys
    int threadsPerBlock = 256;
    int blocksPerGrid = (numKeys + threadsPerBlock - 1) / threadsPerBlock;
    insertKeys<<<blocksPerGrid, threadsPerBlock>>>(d_keys, d_hashTable, numKeys);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch insertKeys kernel" << std::endl;
        return -1;
    }

    // Launch kernel to search keys
    searchKeys<<<blocksPerGrid, threadsPerBlock>>>(d_keys, d_hashTable, d_results, numKeys);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch searchKeys kernel" << std::endl;
        return -1;
    }

    // Copy results from device to host
    err = hipMemcpy(h_results, d_results, numKeys * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy results from device" << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_hashTable);
    hipFree(d_results);

    // Free host memory
    free(h_keys);
    free(h_hashTable);
    free(h_results);

    std::cout << "Hash table operations completed successfully." << std::endl;
    return 0;
}