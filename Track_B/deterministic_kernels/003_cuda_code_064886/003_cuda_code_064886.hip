#include <iostream>
#include <hip/hip_runtime.h>


#define NUM_BUCKETS 64
#define MAX_LOAD_FACTOR 0.75
#define TABLE_SIZE (NUM_BUCKETS / MAX_LOAD_FACTOR)
#define BLOCK_SIZE 256

struct HashTableEntry {
    int key;
    int value;
    bool occupied;
};

__device__ HashTableEntry hashTable[NUM_BUCKETS];

__device__ int hashFunction(int key) {
    return key % NUM_BUCKETS;
}

__device__ bool insertEntry(int key, int value) {
    int hash = hashFunction(key);
    int i = 0;
    while (i < NUM_BUCKETS) {
        int index = (hash + i * i) % NUM_BUCKETS;
        if (!hashTable[index].occupied) {
            hashTable[index].key = key;
            hashTable[index].value = value;
            hashTable[index].occupied = true;
            return true;
        }
        i++;
    }
    return false; // Table is full
}

__device__ int findEntry(int key) {
    int hash = hashFunction(key);
    int i = 0;
    while (i < NUM_BUCKETS) {
        int index = (hash + i * i) % NUM_BUCKETS;
        if (hashTable[index].occupied && hashTable[index].key == key) {
            return hashTable[index].value;
        }
        i++;
    }
    return -1; // Key not found
}

__global__ void initializeHashTable() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_BUCKETS) {
        hashTable[idx].occupied = false;
    }
}

__global__ void insertKeys(int *keys, int *values, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        insertEntry(keys[idx], values[idx]);
    }
}

__global__ void findKeys(int *keys, int *results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        results[idx] = findEntry(keys[idx]);
    }
}

int main() {
    int n = 1024; // Number of keys to insert
    int *h_keys = new int[n];
    int *h_values = new int[n];
    int *h_results = new int[n];

    // Initialize keys and values
    for (int i = 0; i < n; ++i) {
        h_keys[i] = i;
        h_values[i] = i * 10;
        h_results[i] = -1;
    }

    int *d_keys, *d_values, *d_results;
    hipMalloc(&d_keys, n * sizeof(int));
    hipMalloc(&d_values, n * sizeof(int));
    hipMalloc(&d_results, n * sizeof(int));

    // Asynchronous memory copy to device
    hipMemcpyAsync(d_keys, h_keys, n * sizeof(int), hipMemcpyHostToDevice, 0);
    hipMemcpyAsync(d_values, h_values, n * sizeof(int), hipMemcpyHostToDevice, 0);

    // Initialize hash table
    initializeHashTable<<<(NUM_BUCKETS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
    hipDeviceSynchronize();

    // Insert keys into hash table
    insertKeys<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_keys, d_values, n);
    hipDeviceSynchronize();

    // Asynchronous memory copy to device for keys to find
    hipMemcpyAsync(d_keys, h_keys, n * sizeof(int), hipMemcpyHostToDevice, 0);

    // Find keys in hash table
    findKeys<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_keys, d_results, n);
    hipDeviceSynchronize();

    // Asynchronous memory copy back to host
    hipMemcpyAsync(h_results, d_results, n * sizeof(int), hipMemcpyDeviceToHost, 0);
    hipDeviceSynchronize();

    // Verify results
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (h_results[i] != h_values[i]) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "All keys found correctly." << std::endl;
    } else {
        std::cout << "Error in finding keys." << std::endl;
    }

    // Clean up
    delete[] h_keys;
    delete[] h_values;
    delete[] h_results;
    hipFree(d_keys);
    hipFree(d_values);
    hipFree(d_results);

    return 0;
}