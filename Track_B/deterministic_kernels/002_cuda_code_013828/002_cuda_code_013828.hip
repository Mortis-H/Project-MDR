#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cassert>

namespace cg = cooperative_groups;

#define TABLE_SIZE 4096

// Simple hash functions for double hashing
__device__ inline unsigned int hash1(unsigned int key) {
    return key % TABLE_SIZE;
}

__device__ inline unsigned int hash2(unsigned int key) {
    return 1 + (key % (TABLE_SIZE - 1));
}

// Structure to represent a bucket in the hash table
struct Bucket {
    unsigned int key;
    unsigned int value;
    bool occupied;
};

// Kernel to insert elements into the hash table using double hashing
__global__ void insertKernel(Bucket* table, unsigned int* keys, unsigned int* values, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    unsigned int key = keys[idx];
    unsigned int value = values[idx];
    unsigned int h1 = hash1(key);
    unsigned int h2 = hash2(key);

    // Use cooperative groups for warp-level synchronization
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

    // Double hashing to find an empty slot
    for (int i = 0; i < TABLE_SIZE; ++i) {
        unsigned int pos = (h1 + i * h2) % TABLE_SIZE;
        // Load bucket data into shared memory for warp-level access
        __shared__ Bucket sharedBucket;
        if (tile.thread_rank() == 0) {
            sharedBucket = table[pos];
        }
        tile.sync();

        // Check if the bucket is empty or if it contains the same key
        if (!sharedBucket.occupied || sharedBucket.key == key) {
            // Attempt to claim the bucket
            if (atomicCAS((unsigned int*)&table[pos].occupied, 0, 1) == 0) {
                table[pos].key = key;
                table[pos].value = value;
                break;
            }
        }
        tile.sync();
    }
}

// Kernel to search for elements in the hash table
__global__ void searchKernel(Bucket* table, unsigned int* keys, unsigned int* results, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    unsigned int key = keys[idx];
    unsigned int h1 = hash1(key);
    unsigned int h2 = hash2(key);

    // Use cooperative groups for warp-level synchronization
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

    // Double hashing to find the key
    for (int i = 0; i < TABLE_SIZE; ++i) {
        unsigned int pos = (h1 + i * h2) % TABLE_SIZE;
        // Load bucket data into shared memory for warp-level access
        __shared__ Bucket sharedBucket;
        if (tile.thread_rank() == 0) {
            sharedBucket = table[pos];
        }
        tile.sync();

        // Check if the bucket contains the key
        if (sharedBucket.occupied && sharedBucket.key == key) {
            results[idx] = sharedBucket.value;
            break;
        }
        // Mark as not found if we've gone through all buckets
        if (i == TABLE_SIZE - 1) {
            results[idx] = UINT_MAX; // Special value indicating not found
        }
        tile.sync();
    }
}

int main() {
    const int numElements = 1024;
    const int numBlocks = (numElements + 255) / 256;
    const int blockSize = 256;

    // Allocate and initialize host data
    unsigned int* h_keys = new unsigned int[numElements];
    unsigned int* h_values = new unsigned int[numElements];
    unsigned int* h_results = new unsigned int[numElements];
    for (int i = 0; i < numElements; ++i) {
        h_keys[i] = i * 17; // Some arbitrary key generation
        h_values[i] = i;
        h_results[i] = UINT_MAX;
    }

    // Allocate device memory
    unsigned int* d_keys, *d_values, *d_results;
    Bucket* d_table;
    hipMalloc(&d_keys, numElements * sizeof(unsigned int));
    hipMalloc(&d_values, numElements * sizeof(unsigned int));
    hipMalloc(&d_results, numElements * sizeof(unsigned int));
    hipMalloc(&d_table, TABLE_SIZE * sizeof(Bucket));

    // Initialize device table to empty
    hipMemset(d_table, 0, TABLE_SIZE * sizeof(Bucket));

    // Copy data to device
    hipMemcpy(d_keys, h_keys, numElements * sizeof(unsigned int), hipMemcpyHostToDevice);
    hipMemcpy(d_values, h_values, numElements * sizeof(unsigned int), hipMemcpyHostToDevice);

    // Launch insert kernel
    insertKernel<<<numBlocks, blockSize>>>(d_table, d_keys, d_values, numElements);
    hipDeviceSynchronize();
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch search kernel
    searchKernel<<<numBlocks, blockSize>>>(d_table, d_keys, d_results, numElements);
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    hipMemcpy(h_results, d_results, numElements * sizeof(unsigned int), hipMemcpyDeviceToHost);

    // Verify results
    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        if (h_results[i] != h_values[i]) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << h_values[i] << ", got " << h_results[i] << std::endl;
        }
    }
    if (success) {
        std::cout << "All elements found correctly." << std::endl;
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_values);
    hipFree(d_results);
    hipFree(d_table);

    // Free host memory
    delete[] h_keys;
    delete[] h_values;
    delete[] h_results;

    return 0;
}