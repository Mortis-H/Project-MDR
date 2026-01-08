#include <iostream>
#include <hip/hip_runtime.h>

#define TABLE_SIZE 16384
#define MAX_KEY 100000

__device__ int hash_table[TABLE_SIZE];
__device__ int hash_table_keys[TABLE_SIZE];

// Hash function to compute the bucket index
__device__ inline int hash(int key) {
    return key % TABLE_SIZE;
}

// Quadratic probing function to find the next bucket index
__device__ inline int quadratic_probe(int index, int i) {
    return (index + i * i) % TABLE_SIZE;
}

// Kernel to insert keys into the hash table using quadratic probing
__global__ void insert_keys(int *keys, int num_keys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        int key = keys[idx];
        int hash_index = hash(key);
        int i = 0;
        while (true) {
            int next_index = quadratic_probe(hash_index, i);
            int old = atomicCAS(&hash_table[next_index], -1, key);
            if (old == -1) {
                hash_table_keys[next_index] = key;
                break;
            }
            i++;
        }
    }
}

// Kernel to search for keys in the hash table using quadratic probing
__global__ void search_keys(int *keys, int num_keys, int *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        int key = keys[idx];
        int hash_index = hash(key);
        int i = 0;
        while (true) {
            int next_index = quadratic_probe(hash_index, i);
            if (hash_table[next_index] == key) {
                results[idx] = 1; // Key found
                break;
            } else if (hash_table[next_index] == -1) {
                results[idx] = 0; // Key not found
                break;
            }
            i++;
        }
    }
}

int main() {
    // Initialize device memory
    hipError_t err;

    int *d_keys, *d_results;
    int h_keys[TABLE_SIZE], h_results[TABLE_SIZE];

    // Initialize host keys with random values
    for (int i = 0; i < TABLE_SIZE; ++i) {
        h_keys[i] = rand() % MAX_KEY;
        h_results[i] = -1;
    }

    // Allocate device memory
    err = hipMalloc((void **)&d_keys, TABLE_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_results, TABLE_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device results: " << hipGetErrorString(err) << std::endl;
        hipFree(d_keys);
        return -1;
    }

    // Initialize device hash table with -1 (empty)
    err = hipMemset(hash_table, -1, TABLE_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize hash table: " << hipGetErrorString(err) << std::endl;
        hipFree(d_keys);
        hipFree(d_results);
        return -1;
    }

    // Copy keys to device
    err = hipMemcpy(d_keys, h_keys, TABLE_SIZE * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy keys to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_keys);
        hipFree(d_results);
        return -1;
    }

    // Launch insert kernel
    int blockSize = 256;
    int numBlocks = (TABLE_SIZE + blockSize - 1) / blockSize;
    insert_keys<<<numBlocks, blockSize>>>(d_keys, TABLE_SIZE);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch insert kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_keys);
        hipFree(d_results);
        return -1;
    }

    // Launch search kernel
    search_keys<<<numBlocks, blockSize>>>(d_keys, TABLE_SIZE, d_results);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch search kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_keys);
        hipFree(d_results);
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_results, d_results, TABLE_SIZE * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy results to host: " << hipGetErrorString(err) << std::endl;
        hipFree(d_keys);
        hipFree(d_results);
        return -1;
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_results);

    // Verify results
    bool success = true;
    for (int i = 0; i < TABLE_SIZE; ++i) {
        if (h_results[i] != 1) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "All keys found successfully in the hash table." << std::endl;
    } else {
        std::cout << "Error: Some keys were not found in the hash table." << std::endl;
    }

    return 0;
}