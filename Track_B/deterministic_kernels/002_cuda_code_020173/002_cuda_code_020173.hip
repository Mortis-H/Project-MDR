#include <iostream>
#include <hip/hip_runtime.h>

#define BUCKET_SIZE 16
#define NUM_BUCKETS 4096
#define TABLE_SIZE (NUM_BUCKETS * BUCKET_SIZE)

// Hash functions for double hashing
__device__ unsigned int hash1(unsigned int key) {
    return key % NUM_BUCKETS;
}

__device__ unsigned int hash2(unsigned int key) {
    return 1 + (key % (NUM_BUCKETS - 1));
}

// Insert key into hash table using double hashing
__global__ void insertKernel(unsigned int *d_table, unsigned int *d_keys, int num_keys) {
    extern __shared__ unsigned int s_table[];

    // Load data into shared memory
    int idx = threadIdx.x;
    if (idx < TABLE_SIZE) {
        s_table[idx] = d_table[idx];
    }
    __syncthreads();

    // Insert keys into hash table
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_keys; i += blockDim.x * gridDim.x) {
        unsigned int key = d_keys[i];
        unsigned int h1 = hash1(key);
        unsigned int h2 = hash2(key);
        bool inserted = false;

        for (int j = 0; j < NUM_BUCKETS && !inserted; ++j) {
            unsigned int pos = (h1 + j * h2) % NUM_BUCKETS;
            unsigned int bucket_start = pos * BUCKET_SIZE;

            for (int k = 0; k < BUCKET_SIZE; ++k) {
                unsigned int &entry = s_table[bucket_start + k];
                if (entry == 0) {
                    atomicExch(&entry, key);
                    inserted = true;
                    break;
                }
            }
        }
    }

    // Write back to global memory
    if (idx < TABLE_SIZE) {
        d_table[idx] = s_table[idx];
    }
}

// Check if a key exists in the hash table
__global__ void searchKernel(unsigned int *d_table, unsigned int *d_keys, int num_keys, bool *d_results) {
    extern __shared__ unsigned int s_table[];

    // Load data into shared memory
    int idx = threadIdx.x;
    if (idx < TABLE_SIZE) {
        s_table[idx] = d_table[idx];
    }
    __syncthreads();

    // Search for keys in hash table
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_keys; i += blockDim.x * gridDim.x) {
        unsigned int key = d_keys[i];
        unsigned int h1 = hash1(key);
        unsigned int h2 = hash2(key);
        bool found = false;

        for (int j = 0; j < NUM_BUCKETS && !found; ++j) {
            unsigned int pos = (h1 + j * h2) % NUM_BUCKETS;
            unsigned int bucket_start = pos * BUCKET_SIZE;

            for (int k = 0; k < BUCKET_SIZE; ++k) {
                unsigned int entry = s_table[bucket_start + k];
                if (entry == key) {
                    found = true;
                    break;
                }
            }
        }

        d_results[i] = found;
    }
}

int main() {
    const int num_keys = 1024;
    unsigned int *h_keys = new unsigned int[num_keys];
    bool *h_results = new bool[num_keys];
    unsigned int *d_keys, *d_table;
    bool *d_results;

    // Initialize keys
    for (int i = 0; i < num_keys; ++i) {
        h_keys[i] = i * 13; // Simple key generation
        h_results[i] = false;
    }

    // Allocate memory on device
    hipError_t err = hipMalloc(&d_keys, num_keys * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&d_table, TABLE_SIZE * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device table: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&d_results, num_keys * sizeof(bool));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device results: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize table to zero
    hipMemset(d_table, 0, TABLE_SIZE * sizeof(unsigned int));

    // Copy keys to device
    err = hipMemcpy(d_keys, h_keys, num_keys * sizeof(unsigned int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy keys to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Insert keys into hash table
    int blockSize = 256;
    int numBlocks = (num_keys + blockSize - 1) / blockSize;
    insertKernel<<<numBlocks, blockSize, TABLE_SIZE * sizeof(unsigned int)>>>(d_table, d_keys, num_keys);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Search for keys in hash table
    searchKernel<<<numBlocks, blockSize, TABLE_SIZE * sizeof(unsigned int)>>>(d_table, d_keys, num_keys, d_results);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_results, d_results, num_keys * sizeof(bool), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy results to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Verify results
    bool all_found = true;
    for (int i = 0; i < num_keys; ++i) {
        if (!h_results[i]) {
            all_found = false;
            break;
        }
    }

    std::cout << "All keys found: " << (all_found ? "Yes" : "No") << std::endl;

    // Free device memory
    hipFree(d_keys);
    hipFree(d_table);
    hipFree(d_results);

    // Free host memory
    delete[] h_keys;
    delete[] h_results;

    return 0;
}