#include <iostream>
#include <hip/hip_runtime.h>

#define BUCKET_COUNT 512
#define MAX_THREADS_PER_BLOCK 128
#define TABLE_SIZE (BUCKET_COUNT * 4) // Assuming each bucket can hold up to 4 elements

__device__ int hash_table[TABLE_SIZE];

// Kernel to insert elements into the hash table using linear probing
__global__ void insert_kernel(int *keys, int *values, int num_elements) {
    extern __shared__ int shared_keys[];
    extern __shared__ int shared_values[];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;

    // Load data into shared memory
    if (block_offset + tid < num_elements) {
        shared_keys[tid] = keys[block_offset + tid];
        shared_values[tid] = values[block_offset + tid];
    } else {
        shared_keys[tid] = -1; // Mark invalid keys
    }
    __syncthreads();

    // Insert elements into the hash table
    if (shared_keys[tid] != -1) {
        int hash = shared_keys[tid] % BUCKET_COUNT;
        int index = hash * 4; // Start of the bucket

        while (true) {
            // Linear probing
            for (int i = 0; i < 4; i++) {
                int pos = index + i;
                if (atomicCAS(&hash_table[pos], 0, shared_keys[tid]) == 0) {
                    hash_table[pos + TABLE_SIZE / 4] = shared_values[tid];
                    break;
                }
            }
            // If bucket is full, move to the next bucket
            hash = (hash + 1) % BUCKET_COUNT;
            index = hash * 4;
        }
    }
}

// Kernel to search for elements in the hash table
__global__ void search_kernel(int *keys, int *results, int num_elements) {
    extern __shared__ int shared_keys[];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;

    // Load data into shared memory
    if (block_offset + tid < num_elements) {
        shared_keys[tid] = keys[block_offset + tid];
    } else {
        shared_keys[tid] = -1; // Mark invalid keys
    }
    __syncthreads();

    // Search for elements in the hash table
    if (shared_keys[tid] != -1) {
        int hash = shared_keys[tid] % BUCKET_COUNT;
        int index = hash * 4; // Start of the bucket

        bool found = false;
        for (int i = 0; i < 4 && !found; i++) {
            int pos = index + i;
            if (hash_table[pos] == shared_keys[tid]) {
                results[block_offset + tid] = hash_table[pos + TABLE_SIZE / 4];
                found = true;
            }
        }

        // If not found in the current bucket, continue searching
        if (!found) {
            hash = (hash + 1) % BUCKET_COUNT;
            index = hash * 4;
            while (hash_table[index] != 0 && !found) {
                for (int i = 0; i < 4 && !found; i++) {
                    int pos = index + i;
                    if (hash_table[pos] == shared_keys[tid]) {
                        results[block_offset + tid] = hash_table[pos + TABLE_SIZE / 4];
                        found = true;
                    }
                }
                hash = (hash + 1) % BUCKET_COUNT;
                index = hash * 4;
            }
        }

        // If not found, set result to -1
        if (!found) {
            results[block_offset + tid] = -1;
        }
    }
}

int main() {
    // Initialize CUDA
    hipError_t err;

    // Example data
    int num_elements = 1024;
    int *h_keys = new int[num_elements];
    int *h_values = new int[num_elements];
    int *h_results = new int[num_elements];

    for (int i = 0; i < num_elements; i++) {
        h_keys[i] = i; // Unique keys for simplicity
        h_values[i] = i * 10; // Arbitrary values
        h_results[i] = -1; // Initialize results
    }

    // Allocate device memory
    int *d_keys, *d_values, *d_results;
    err = hipMalloc((void **)&d_keys, num_elements * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device keys memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_values, num_elements * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device values memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_results, num_elements * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device results memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy data to device
    err = hipMemcpy(d_keys, h_keys, num_elements * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy keys to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMemcpy(d_values, h_values, num_elements * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy values to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize hash table
    err = hipMemset(hash_table, 0, TABLE_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize hash table: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch insert kernel
    int num_blocks = (num_elements + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    insert_kernel<<<num_blocks, MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK * sizeof(int) * 2>>>(d_keys, d_values, num_elements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Insert kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch search kernel
    search_kernel<<<num_blocks, MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK * sizeof(int)>>>(d_keys, d_results, num_elements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Search kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_results, d_results, num_elements * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy results from device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Verify results
    bool success = true;
    for (int i = 0; i < num_elements; i++) {
        if (h_results[i] != h_values[i]) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << h_values[i] << ", got " << h_results[i] << std::endl;
        }
    }
    if (success) {
        std::cout << "All elements found correctly!" << std::endl;
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_values);
    hipFree(d_results);

    // Free host memory
    delete[] h_keys;
    delete[] h_values;
    delete[] h_results;

    return 0;
}