#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

#define BUCKETS 64
#define TABLE_SIZE (BUCKETS * 2)

__device__ unsigned int hash_function(unsigned int key) {
    return key % BUCKETS;
}

__device__ unsigned int quadratic_probe(unsigned int hash, unsigned int i) {
    return (hash + i * i) % BUCKETS;
}

__global__ void insert_keys(unsigned int *keys, unsigned int *table, int num_keys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;

    unsigned int key = keys[idx];
    unsigned int hash = hash_function(key);
    unsigned int i = 0;

    cg::thread_block cta = cg::this_thread_block();
    while (i < BUCKETS) {
        unsigned int slot = quadratic_probe(hash, i);
        unsigned int table_index = slot * 2;

        // Use atomic operations to ensure no race conditions
        unsigned int old = atomicCAS(&table[table_index], 0, key);
        if (old == 0) {
            // Insertion successful
            table[table_index + 1] = 1; // Mark as occupied
            break;
        }
        i++;
        cg::sync(cta); // Ensure all threads are synchronized
    }
}

__global__ void search_keys(unsigned int *keys, unsigned int *table, int num_keys, int *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;

    unsigned int key = keys[idx];
    unsigned int hash = hash_function(key);
    unsigned int i = 0;

    cg::thread_block cta = cg::this_thread_block();
    while (i < BUCKETS) {
        unsigned int slot = quadratic_probe(hash, i);
        unsigned int table_index = slot * 2;

        if (table[table_index] == key) {
            results[idx] = 1; // Key found
            break;
        } else if (table[table_index + 1] == 0) {
            results[idx] = 0; // Key not found
            break;
        }
        i++;
        cg::sync(cta); // Ensure all threads are synchronized
    }
}

int main() {
    const int num_keys = 1024;
    unsigned int *h_keys = new unsigned int[num_keys];
    int *h_results = new int[num_keys];

    // Initialize keys with some values
    for (int i = 0; i < num_keys; i++) {
        h_keys[i] = i * 3; // Simple key generation
        h_results[i] = -1; // Initialize results to -1
    }

    unsigned int *d_keys, *d_table;
    int *d_results;
    hipError_t err;

    // Allocate memory on the device
    err = hipMalloc(&d_keys, num_keys * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device keys" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_table, TABLE_SIZE * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device table" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_results, num_keys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device results" << std::endl;
        return -1;
    }

    // Copy keys to the device
    err = hipMemcpy(d_keys, h_keys, num_keys * sizeof(unsigned int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy keys to device" << std::endl;
        return -1;
    }

    // Set up the execution configuration
    int threads_per_block = 256;
    int blocks_per_grid = (num_keys + threads_per_block - 1) / threads_per_block;

    // Launch the insert kernel
    insert_keys<<<blocks_per_grid, threads_per_block>>>(d_keys, d_table, num_keys);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch insert kernel" << std::endl;
        return -1;
    }
    hipDeviceSynchronize();

    // Launch the search kernel
    search_keys<<<blocks_per_grid, threads_per_block>>>(d_keys, d_table, num_keys, d_results);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch search kernel" << std::endl;
        return -1;
    }
    hipDeviceSynchronize();

    // Copy results back to the host
    err = hipMemcpy(h_results, d_results, num_keys * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy results from device" << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_table);
    hipFree(d_results);

    // Clean up host memory
    delete[] h_keys;
    delete[] h_results;

    return 0;
}