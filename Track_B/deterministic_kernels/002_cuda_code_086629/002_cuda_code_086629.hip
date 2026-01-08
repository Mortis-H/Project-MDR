#include <iostream>
#include <hip/hip_runtime.h>

#define TABLE_SIZE 4096

// Hash function for simplicity, not cryptographic
__device__ unsigned int hash(unsigned int key) {
    return key % TABLE_SIZE;
}

// Kernel to insert keys into the hash table
__global__ void insertKernel(unsigned int *keys, unsigned int *table, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys) {
        unsigned int h = hash(keys[idx]);
        while (atomicCAS(&table[h], 0, keys[idx]) != 0) {
            h = (h + 1) % TABLE_SIZE; // Linear probing
        }
    }
}

// Kernel to search for keys in the hash table
__global__ void searchKernel(unsigned int *keys, unsigned int *table, int numKeys, int *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys) {
        unsigned int h = hash(keys[idx]);
        while (table[h] != keys[idx] && table[h] != 0) {
            h = (h + 1) % TABLE_SIZE; // Linear probing
        }
        results[idx] = (table[h] == keys[idx]) ? h : -1; // -1 if not found
    }
}

int main() {
    const int numKeys = 2048;
    unsigned int *h_keys, *h_table, *d_keys, *d_table;
    int *h_results, *d_results;

    // Allocate host memory
    h_keys = (unsigned int *)malloc(numKeys * sizeof(unsigned int));
    h_table = (unsigned int *)calloc(TABLE_SIZE, sizeof(unsigned int));
    h_results = (int *)malloc(numKeys * sizeof(int));

    // Allocate device memory
    hipError_t err = hipMalloc((void **)&d_keys, numKeys * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_table, TABLE_SIZE * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_table: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_results, numKeys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_results: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize host keys
    for (int i = 0; i < numKeys; i++) {
        h_keys[i] = i * 2; // Simple key generation
    }

    // Asynchronously copy keys to device
    err = hipMemcpyAsync(d_keys, h_keys, numKeys * sizeof(unsigned int), hipMemcpyHostToDevice, 0);
    if (err != hipSuccess) {
        std::cerr << "Error copying h_keys to d_keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Set grid and block dimensions
    dim3 blockSize(256);
    dim3 gridSize((numKeys + blockSize.x - 1) / blockSize.x);

    // Launch insert kernel
    insertKernel<<<gridSize, blockSize>>>(d_keys, d_table, numKeys);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching insertKernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Asynchronously copy table to host for verification (not necessary for actual use)
    err = hipMemcpyAsync(h_table, d_table, TABLE_SIZE * sizeof(unsigned int), hipMemcpyDeviceToHost, 0);
    if (err != hipSuccess) {
        std::cerr << "Error copying d_table to h_table: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch search kernel
    searchKernel<<<gridSize, blockSize>>>(d_keys, d_table, numKeys, d_results);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching searchKernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Asynchronously copy results to host
    err = hipMemcpyAsync(h_results, d_results, numKeys * sizeof(int), hipMemcpyDeviceToHost, 0);
    if (err != hipSuccess) {
        std::cerr << "Error copying d_results to h_results: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Synchronize to ensure all operations are complete
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Error during device synchronization: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Verify results
    bool success = true;
    for (int i = 0; i < numKeys; i++) {
        if (h_results[i] == -1 || h_table[h_results[i]] != h_keys[i]) {
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Hash table operations completed successfully." << std::endl;
    } else {
        std::cout << "Hash table operations failed." << std::endl;
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_table);
    hipFree(d_results);

    // Free host memory
    free(h_keys);
    free(h_table);
    free(h_results);

    return 0;
}