#include <iostream>
#include <hip/hip_runtime.h>


#define NUM_BUCKETS 16384
#define HASH_TABLE_SIZE (NUM_BUCKETS * 2) // To avoid collisions

__device__ int hash1(int key) {
    return key % NUM_BUCKETS;
}

__device__ int hash2(int key) {
    return 1 + (key % (NUM_BUCKETS - 1));
}

__global__ void insert_kernel(int* d_keys, int* d_hash_table, int num_keys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        int key = d_keys[idx];
        int h1 = hash1(key);
        int h2 = hash2(key);
        int i = 0;
        int pos;
        do {
            pos = (h1 + i * h2) % HASH_TABLE_SIZE;
            i++;
        } while (atomicCAS(&d_hash_table[pos], 0, key) != 0);
    }
}

__global__ void search_kernel(int* d_keys, int* d_hash_table, int num_keys, int* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        int key = d_keys[idx];
        int h1 = hash1(key);
        int h2 = hash2(key);
        int i = 0;
        int pos;
        do {
            pos = (h1 + i * h2) % HASH_TABLE_SIZE;
            if (d_hash_table[pos] == key) {
                d_results[idx] = pos;
                return;
            }
            i++;
        } while (d_hash_table[pos] != 0);
        d_results[idx] = -1; // Key not found
    }
}

void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int num_keys = 1024;
    int* h_keys = new int[num_keys];
    for (int i = 0; i < num_keys; i++) {
        h_keys[i] = i * 17; // Generate some keys
    }

    int* d_keys;
    int* d_hash_table;
    int* d_results;
    checkCudaError(hipMalloc(&d_keys, num_keys * sizeof(int)), "hipMalloc d_keys");
    checkCudaError(hipMalloc(&d_hash_table, HASH_TABLE_SIZE * sizeof(int)), "hipMalloc d_hash_table");
    checkCudaError(hipMalloc(&d_results, num_keys * sizeof(int)), "hipMalloc d_results");

    checkCudaError(hipMemset(d_hash_table, 0, HASH_TABLE_SIZE * sizeof(int)), "hipMemset d_hash_table");
    checkCudaError(hipMemcpy(d_keys, h_keys, num_keys * sizeof(int), hipMemcpyHostToDevice), "hipMemcpy d_keys");

    int blockSize = 256;
    int numBlocks = (num_keys + blockSize - 1) / blockSize;

    // Insert keys into hash table
    insert_kernel<<<numBlocks, blockSize>>>(d_keys, d_hash_table, num_keys);
    checkCudaError(hipGetLastError(), "insert_kernel launch failure");
    checkCudaError(hipDeviceSynchronize(), "hipDeviceSynchronize after insert_kernel");

    // Search keys in hash table
    search_kernel<<<numBlocks, blockSize>>>(d_keys, d_hash_table, num_keys, d_results);
    checkCudaError(hipGetLastError(), "search_kernel launch failure");
    checkCudaError(hipDeviceSynchronize(), "hipDeviceSynchronize after search_kernel");

    // Copy results back to host
    int* h_results = new int[num_keys];
    checkCudaError(hipMemcpy(h_results, d_results, num_keys * sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy h_results");

    // Free device memory
    hipFree(d_keys);
    hipFree(d_hash_table);
    hipFree(d_results);

    // Free host memory
    delete[] h_keys;
    delete[] h_results;

    std::cout << "Hash table operations completed successfully." << std::endl;

    return 0;
}