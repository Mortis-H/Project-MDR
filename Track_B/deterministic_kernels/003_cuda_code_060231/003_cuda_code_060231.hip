#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define BUCKETS 64
#define MAX_CHAIN_LENGTH 1024
#define MAX_THREADS 256

__device__ int hash_table[BUCKETS][MAX_CHAIN_LENGTH];
__device__ int chain_lengths[BUCKETS];

__global__ void initializeHashTable() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BUCKETS) {
        chain_lengths[idx] = 0;
        for (int i = 0; i < MAX_CHAIN_LENGTH; ++i) {
            hash_table[idx][i] = -1; // -1 indicates an empty slot
        }
    }
}

__global__ void insertIntoHashTable(int *keys, int num_keys) {
    extern __shared__ int shared_keys[];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    if (global_idx < num_keys) {
        shared_keys[tid] = keys[global_idx];
        __syncthreads();

        int key = shared_keys[tid];
        int bucket = key % BUCKETS;

        int pos = atomicAdd(&chain_lengths[bucket], 1);
        if (pos < MAX_CHAIN_LENGTH) {
            hash_table[bucket][pos] = key;
        }
    }
}

__global__ void searchHashTable(int *keys, int num_keys, int *results) {
    extern __shared__ int shared_keys[];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    if (global_idx < num_keys) {
        shared_keys[tid] = keys[global_idx];
        __syncthreads();

        int key = shared_keys[tid];
        int bucket = key % BUCKETS;
        bool found = false;

        for (int i = 0; i < chain_lengths[bucket] && !found; ++i) {
            if (hash_table[bucket][i] == key) {
                results[global_idx] = 1;
                found = true;
            }
        }
        if (!found) {
            results[global_idx] = 0;
        }
    }
}

int main() {
    int num_keys = 1024;
    thrust::host_vector<int> h_keys(num_keys);
    thrust::host_vector<int> h_results(num_keys);

    for (int i = 0; i < num_keys; ++i) {
        h_keys[i] = i; // Simple keys for demonstration
    }

    thrust::device_vector<int> d_keys = h_keys;
    thrust::device_vector<int> d_results(num_keys);

    int *d_keys_ptr = thrust::raw_pointer_cast(d_keys.data());
    int *d_results_ptr = thrust::raw_pointer_cast(d_results.data());

    int num_blocks = (num_keys + MAX_THREADS - 1) / MAX_THREADS;

    // Initialize hash table
    initializeHashTable<<<(BUCKETS + MAX_THREADS - 1) / MAX_THREADS, MAX_THREADS>>>();
    hipDeviceSynchronize();

    // Insert keys into hash table
    insertIntoHashTable<<<num_blocks, MAX_THREADS, MAX_THREADS * sizeof(int)>>>(d_keys_ptr, num_keys);
    hipDeviceSynchronize();

    // Search keys in hash table
    searchHashTable<<<num_blocks, MAX_THREADS, MAX_THREADS * sizeof(int)>>>(d_keys_ptr, num_keys, d_results_ptr);
    hipDeviceSynchronize();

    // Copy results back to host
    h_results = d_results;

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Output results
    for (int i = 0; i < num_keys; ++i) {
        if (h_results[i] != 1) {
            std::cerr << "Key " << h_keys[i] << " not found!" << std::endl;
            return -1;
        }
    }

    std::cout << "All keys found successfully!" << std::endl;

    return 0;
}