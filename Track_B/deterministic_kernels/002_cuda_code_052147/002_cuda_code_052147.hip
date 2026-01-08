#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_BUCKETS 1024
#define MAX_THREADS_PER_BLOCK 256

// Simple hash function for demonstration purposes
__device__ unsigned int hashFunction(unsigned int key) {
    return key % NUM_BUCKETS;
}

// Quadratic probing function
__device__ unsigned int quadraticProbe(unsigned int i) {
    return (i * i + i) / 2;
}

// Hash table structure
struct HashTable {
    int *keys;
    int *values;
};

// Kernel to insert keys into the hash table using quadratic probing
__global__ void insertKeys(HashTable ht, int *keys, int *values, int numKeys) {
    extern __shared__ int sharedKeys[];
    extern __shared__ int sharedValues[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numKeys) return;

    // Load data into shared memory
    sharedKeys[threadIdx.x] = keys[idx];
    sharedValues[threadIdx.x] = values[idx];
    __syncthreads();

    // Insert key into hash table
    unsigned int key = sharedKeys[threadIdx.x];
    unsigned int value = sharedValues[threadIdx.x];
    unsigned int bucket = hashFunction(key);
    unsigned int i = 0;

    while (true) {
        unsigned int finalBucket = (bucket + quadraticProbe(i)) % NUM_BUCKETS;
        int oldKey = atomicCAS(&ht.keys[finalBucket], 0, key);
        if (oldKey == 0) {
            ht.values[finalBucket] = value;
            break;
        } else if (oldKey == key) {
            // Key already exists, update value
            ht.values[finalBucket] = value;
            break;
        }
        i++;
    }
}

// Kernel to search for keys in the hash table
__global__ void searchKeys(HashTable ht, int *searchKeys, int *results, int numSearchKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSearchKeys) return;

    unsigned int key = searchKeys[idx];
    unsigned int bucket = hashFunction(key);
    unsigned int i = 0;

    while (true) {
        unsigned int finalBucket = (bucket + quadraticProbe(i)) % NUM_BUCKETS;
        if (ht.keys[finalBucket] == key) {
            results[idx] = ht.values[finalBucket];
            break;
        } else if (ht.keys[finalBucket] == 0) {
            results[idx] = -1; // Key not found
            break;
        }
        i++;
    }
}

int main() {
    const int numKeys = 512;
    const int numSearchKeys = 256;
    int *h_keys, *h_values, *h_searchKeys, *h_results;
    int *d_keys, *d_values, *d_searchKeys, *d_results;

    // Allocate host memory
    h_keys = (int *)malloc(numKeys * sizeof(int));
    h_values = (int *)malloc(numKeys * sizeof(int));
    h_searchKeys = (int *)malloc(numSearchKeys * sizeof(int));
    h_results = (int *)malloc(numSearchKeys * sizeof(int));

    // Initialize host arrays
    for (int i = 0; i < numKeys; i++) {
        h_keys[i] = i;
        h_values[i] = i * 10;
    }
    for (int i = 0; i < numSearchKeys; i++) {
        h_searchKeys[i] = i * 2; // Search for even keys
    }

    // Allocate device memory
    hipMalloc(&d_keys, numKeys * sizeof(int));
    hipMalloc(&d_values, NUM_BUCKETS * sizeof(int));
    hipMalloc(&d_searchKeys, numSearchKeys * sizeof(int));
    hipMalloc(&d_results, numSearchKeys * sizeof(int));

    // Initialize device memory
    hipMemset(d_values, 0, NUM_BUCKETS * sizeof(int));

    // Copy data from host to device
    hipMemcpy(d_keys, h_keys, numKeys * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_searchKeys, h_searchKeys, numSearchKeys * sizeof(int), hipMemcpyHostToDevice);

    // Create hash table structure on device
    HashTable d_ht;
    d_ht.keys = d_keys;
    d_ht.values = d_values;

    // Launch kernel to insert keys
    int blockSize = MAX_THREADS_PER_BLOCK;
    int numBlocks = (numKeys + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(int) * 2; // Space for keys and values
    insertKeys<<<numBlocks, blockSize, sharedMemSize>>>(d_ht, d_keys, d_values, numKeys);
    hipDeviceSynchronize();

    // Launch kernel to search for keys
    numBlocks = (numSearchKeys + blockSize - 1) / blockSize;
    searchKeys<<<numBlocks, blockSize>>>(d_ht, d_searchKeys, d_results, numSearchKeys);
    hipDeviceSynchronize();

    // Copy results from device to host
    hipMemcpy(h_results, d_results, numSearchKeys * sizeof(int), hipMemcpyDeviceToHost);

    // Print results
    std::cout << "Search Results:\n";
    for (int i = 0; i < numSearchKeys; i++) {
        std::cout << "Key: " << h_searchKeys[i] << ", Value: " << h_results[i] << "\n";
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_values);
    hipFree(d_searchKeys);
    hipFree(d_results);

    // Free host memory
    free(h_keys);
    free(h_values);
    free(h_searchKeys);
    free(h_results);

    return 0;
}