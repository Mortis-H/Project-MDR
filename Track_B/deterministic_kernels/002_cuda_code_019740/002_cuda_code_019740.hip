#include <iostream>
#include <hip/hip_runtime.h>

#define NUM_BUCKETS 1024
#define MAX_KEYS 2048

// Kernel to insert keys into the hash table using linear probing
__global__ void insertKeys(int* keys, int* hashTable, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys) {
        int key = keys[idx];
        int hash = key % NUM_BUCKETS;
        int i = 0;

        // Linear probing to find an empty slot
        while (i < NUM_BUCKETS) {
            int slot = (hash + i) % NUM_BUCKETS;
            int old = atomicCAS(&hashTable[slot], 0, key);
            if (old == 0) break; // Insertion successful
            i++;
        }
    }
}

// Kernel to search for keys in the hash table using linear probing
__global__ void searchKeys(int* keys, int* hashTable, int* results, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys) {
        int key = keys[idx];
        int hash = key % NUM_BUCKETS;
        int i = 0;

        // Linear probing to find the key
        while (i < NUM_BUCKETS) {
            int slot = (hash + i) % NUM_BUCKETS;
            if (hashTable[slot] == key) {
                results[idx] = 1; // Key found
                break;
            } else if (hashTable[slot] == 0) {
                results[idx] = 0; // Key not found
                break;
            }
            i++;
        }
    }
}

int main() {
    int* h_keys;
    int* d_keys;
    int* d_hashTable;
    int* d_results;

    // Allocate host memory
    h_keys = (int*)malloc(MAX_KEYS * sizeof(int));
    int* h_results = (int*)malloc(MAX_KEYS * sizeof(int));

    // Initialize keys
    for (int i = 0; i < MAX_KEYS; i++) {
        h_keys[i] = i; // Simple key initialization
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_keys, MAX_KEYS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device keys" << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_hashTable, NUM_BUCKETS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device hash table" << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_results, MAX_KEYS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device results" << std::endl;
        return -1;
    }

    // Initialize hash table to 0
    err = hipMemset(d_hashTable, 0, NUM_BUCKETS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize hash table" << std::endl;
        return -1;
    }

    // Copy keys to device
    err = hipMemcpy(d_keys, h_keys, MAX_KEYS * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy keys to device" << std::endl;
        return -1;
    }

    // Launch insert kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (MAX_KEYS + threadsPerBlock - 1) / threadsPerBlock;
    insertKeys<<<blocksPerGrid, threadsPerBlock>>>(d_keys, d_hashTable, MAX_KEYS);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch insert kernel" << std::endl;
        return -1;
    }

    // Launch search kernel
    searchKeys<<<blocksPerGrid, threadsPerBlock>>>(d_keys, d_hashTable, d_results, MAX_KEYS);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch search kernel" << std::endl;
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_results, d_results, MAX_KEYS * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy results to host" << std::endl;
        return -1;
    }

    // Verify results
    bool success = true;
    for (int i = 0; i < MAX_KEYS; i++) {
        if (h_results[i] != 1) {
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "All keys found successfully in the hash table." << std::endl;
    } else {
        std::cout << "Error: Not all keys were found in the hash table." << std::endl;
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_hashTable);
    hipFree(d_results);

    // Free host memory
    free(h_keys);
    free(h_results);

    return 0;
}