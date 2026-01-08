#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define BUCKET_COUNT 16384
#define MAX_THREADS 1024
#define MAX_BLOCKS 64

// Structure to represent a hash table bucket
struct Bucket {
    int key;
    int value;
    bool occupied;
};

// Kernel to initialize hash table buckets
__global__ void initHashTable(Bucket *hashTable) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BUCKET_COUNT) {
        hashTable[idx].occupied = false;
    }
}

// Kernel to insert keys into the hash table using linear probing
__global__ void insertKeys(Bucket *hashTable, int *keys, int *values, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys) {
        int key = keys[idx];
        int value = values[idx];
        int hashIndex = key % BUCKET_COUNT;
        while (hashTable[hashIndex].occupied && hashTable[hashIndex].key != key) {
            hashIndex = (hashIndex + 1) % BUCKET_COUNT;
        }
        if (!hashTable[hashIndex].occupied) {
            hashTable[hashIndex].key = key;
            hashTable[hashIndex].value = value;
            hashTable[hashIndex].occupied = true;
        }
    }
}

// Kernel to search for keys in the hash table
__global__ void searchKeys(Bucket *hashTable, int *searchKeys, int *results, int numSearchKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSearchKeys) {
        int key = searchKeys[idx];
        int hashIndex = key % BUCKET_COUNT;
        while (hashTable[hashIndex].occupied && hashTable[hashIndex].key != key) {
            hashIndex = (hashIndex + 1) % BUCKET_COUNT;
        }
        if (hashTable[hashIndex].occupied && hashTable[hashIndex].key == key) {
            results[idx] = hashTable[hashIndex].value;
        } else {
            results[idx] = -1; // Key not found
        }
    }
}

int main() {
    int numKeys = 8192;
    int numSearchKeys = 1024;
    int *h_keys, *h_values, *h_searchKeys, *h_results;
    int *d_keys, *d_values, *d_searchKeys, *d_results;
    Bucket *d_hashTable;

    // Allocate host memory
    h_keys = (int *)malloc(numKeys * sizeof(int));
    h_values = (int *)malloc(numKeys * sizeof(int));
    h_searchKeys = (int *)malloc(numSearchKeys * sizeof(int));
    h_results = (int *)malloc(numSearchKeys * sizeof(int));

    // Initialize keys and values
    for (int i = 0; i < numKeys; ++i) {
        h_keys[i] = i;
        h_values[i] = i * 2;
    }

    // Initialize search keys
    for (int i = 0; i < numSearchKeys; ++i) {
        h_searchKeys[i] = i + 4096; // Some keys are present, some are not
    }

    // Allocate device memory
    hipMalloc((void **)&d_keys, numKeys * sizeof(int));
    hipMalloc((void **)&d_values, numKeys * sizeof(int));
    hipMalloc((void **)&d_searchKeys, numSearchKeys * sizeof(int));
    hipMalloc((void **)&d_results, numSearchKeys * sizeof(int));
    hipMalloc((void **)&d_hashTable, BUCKET_COUNT * sizeof(Bucket));

    // Copy data from host to device
    hipMemcpy(d_keys, h_keys, numKeys * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_values, h_values, numKeys * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_searchKeys, h_searchKeys, numSearchKeys * sizeof(int), hipMemcpyHostToDevice);

    // Initialize hash table
    initHashTable<<<(BUCKET_COUNT + MAX_THREADS - 1) / MAX_THREADS, MAX_THREADS>>>(d_hashTable);
    hipDeviceSynchronize();
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing hash table: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Insert keys into hash table
    insertKeys<<<(numKeys + MAX_THREADS - 1) / MAX_THREADS, MAX_THREADS>>>(d_hashTable, d_keys, d_values, numKeys);
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error inserting keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Search for keys in hash table
    searchKeys<<<(numSearchKeys + MAX_THREADS - 1) / MAX_THREADS, MAX_THREADS>>>(d_hashTable, d_searchKeys, d_results, numSearchKeys);
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error searching keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results from device to host
    hipMemcpy(h_results, d_results, numSearchKeys * sizeof(int), hipMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < numSearchKeys; ++i) {
        std::cout << "Key: " << h_searchKeys[i] << ", Value: " << h_results[i] << std::endl;
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_values);
    hipFree(d_searchKeys);
    hipFree(d_results);
    hipFree(d_hashTable);

    // Free host memory
    free(h_keys);
    free(h_values);
    free(h_searchKeys);
    free(h_results);

    return 0;
}