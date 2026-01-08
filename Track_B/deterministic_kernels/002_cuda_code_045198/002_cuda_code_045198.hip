#include <iostream>
#include <hip/hip_runtime.h>

#define NUM_BUCKETS 16384
#define HASH_TABLE_SIZE 32768 // Twice the number of buckets for quadratic probing
#define MAX_THREADS 1024

// Kernel to insert keys into the hash table using quadratic probing
__global__ void insertKeys(int* hashTable, int* keys, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numKeys) return;

    int key = keys[idx];
    int hash = key % NUM_BUCKETS;
    int i = 0;
    bool inserted = false;

    // Quadratic probing
    while (i < NUM_BUCKETS && !inserted) {
        int probe = (hash + i * i) % HASH_TABLE_SIZE;
        int old = atomicCAS(&hashTable[probe], 0, key);
        if (old == 0) {
            inserted = true;
        }
        i++;
    }

    if (!inserted) {
        printf("Failed to insert key %d\n", key);
    }
}

// Kernel to search for keys in the hash table using quadratic probing
__global__ void searchKeys(int* hashTable, int* keys, int* results, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numKeys) return;

    int key = keys[idx];
    int hash = key % NUM_BUCKETS;
    int i = 0;
    bool found = false;

    // Quadratic probing
    while (i < NUM_BUCKETS && !found) {
        int probe = (hash + i * i) % HASH_TABLE_SIZE;
        if (hashTable[probe] == key) {
            results[idx] = 1;
            found = true;
        } else if (hashTable[probe] == 0) {
            results[idx] = 0;
            found = true;
        }
        i++;
    }

    if (!found) {
        results[idx] = 0;
    }
}

int main() {
    int numKeys = 1024;
    int* keys = new int[numKeys];
    int* results = new int[numKeys];

    // Initialize keys with some values
    for (int i = 0; i < numKeys; i++) {
        keys[i] = i * 3 + 1; // Example key generation
    }

    int* d_keys;
    int* d_hashTable;
    int* d_results;

    // Allocate memory on the device
    hipError_t err = hipMalloc(&d_keys, numKeys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc(&d_hashTable, HASH_TABLE_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for hash table: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc(&d_results, numKeys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for results: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize hash table to 0
    err = hipMemset(d_hashTable, 0, HASH_TABLE_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize hash table: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy keys to device
    err = hipMemcpy(d_keys, keys, numKeys * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy keys to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch kernel to insert keys
    int threadsPerBlock = 256;
    int blocksPerGrid = (numKeys + threadsPerBlock - 1) / threadsPerBlock;
    insertKeys<<<blocksPerGrid, threadsPerBlock>>>(d_hashTable, d_keys, numKeys);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch insertKeys kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch kernel to search keys
    searchKeys<<<blocksPerGrid, threadsPerBlock>>>(d_hashTable, d_keys, d_results, numKeys);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch searchKeys kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(results, d_results, numKeys * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy results back to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_hashTable);
    hipFree(d_results);

    // Free host memory
    delete[] keys;
    delete[] results;

    std::cout << "Hash table operations completed successfully." << std::endl;
    return 0;
}