#include <iostream>
#include <hip/hip_runtime.h>

#define BUCKETS 2048
#define MAX_LOAD_FACTOR 0.75
#define MAX_KEYS (int)(BUCKETS * MAX_LOAD_FACTOR)

__device__ int hash_table[BUCKETS];
__device__ int keys[MAX_KEYS];
__device__ int values[MAX_KEYS];

__global__ void initializeHashTable() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BUCKETS) {
        hash_table[idx] = -1; // -1 indicates an empty bucket
    }
}

__device__ int hashFunction(int key) {
    return key % BUCKETS;
}

__global__ void insertKeys(int* d_keys, int* d_values, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys) {
        int key = d_keys[idx];
        int value = d_values[idx];
        int hash = hashFunction(key);
        while (hash_table[hash] != -1 && hash_table[hash] != key) {
            hash = (hash + 1) % BUCKETS; // Linear probing
        }
        if (hash_table[hash] == -1) {
            hash_table[hash] = key;
            values[hash] = value;
        }
    }
}

__global__ void searchKeys(int* d_keys, int* d_results, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys) {
        int key = d_keys[idx];
        int hash = hashFunction(key);
        while (hash_table[hash] != -1 && hash_table[hash] != key) {
            hash = (hash + 1) % BUCKETS; // Linear probing
        }
        d_results[idx] = (hash_table[hash] == key) ? values[hash] : -1;
    }
}

int main() {
    int numKeys = MAX_KEYS;
    int* h_keys = new int[numKeys];
    int* h_values = new int[numKeys];
    int* h_results = new int[numKeys];

    // Initialize keys and values
    for (int i = 0; i < numKeys; ++i) {
        h_keys[i] = i;
        h_values[i] = i * 10;
    }

    int* d_keys;
    int* d_values;
    int* d_results;
    hipError_t err;

    err = hipMalloc((void**)&d_keys, numKeys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_values, numKeys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_values: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_results, numKeys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_results: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMemcpy(d_keys, h_keys, numKeys * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying h_keys to d_keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMemcpy(d_values, h_values, numKeys * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying h_values to d_values: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (numKeys + threadsPerBlock - 1) / threadsPerBlock;

    initializeHashTable<<<blocksPerGrid, threadsPerBlock>>>();
    hipDeviceSynchronize();

    insertKeys<<<blocksPerGrid, threadsPerBlock>>>(d_keys, d_values, numKeys);
    hipDeviceSynchronize();

    searchKeys<<<blocksPerGrid, threadsPerBlock>>>(d_keys, d_results, numKeys);
    hipDeviceSynchronize();

    err = hipMemcpy(h_results, d_results, numKeys * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying d_results to h_results: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Verify results
    for (int i = 0; i < numKeys; ++i) {
        if (h_results[i] != h_values[i]) {
            std::cerr << "Mismatch at index " << i << ": expected " << h_values[i] << ", got " << h_results[i] << std::endl;
            return -1;
        }
    }

    std::cout << "All keys found successfully." << std::endl;

    hipFree(d_keys);
    hipFree(d_values);
    hipFree(d_results);
    delete[] h_keys;
    delete[] h_values;
    delete[] h_results;

    return 0;
}