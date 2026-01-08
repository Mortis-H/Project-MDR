#include <iostream>
#include <hip/hip_runtime.h>


#define NUM_BUCKETS 256
#define MAX_KEY 10000
#define BLOCK_SIZE 256

__device__ int hash_table[NUM_BUCKETS];
__device__ int hash_table_status[NUM_BUCKETS]; // 0 = empty, 1 = occupied

__device__ int hash_function(int key) {
    return key % NUM_BUCKETS;
}

__device__ void insert_key(int key) {
    int index = hash_function(key);
    int i = 0;
    while (hash_table_status[index] == 1 && hash_table[index] != key) {
        index = (index + 1) % NUM_BUCKETS;
        i++;
        if (i == NUM_BUCKETS) {
            printf("Hash table full, could not insert key %d\n", key);
            return;
        }
    }
    if (hash_table_status[index] == 0) {
        hash_table[index] = key;
        hash_table_status[index] = 1;
    }
}

__device__ bool search_key(int key) {
    int index = hash_function(key);
    int i = 0;
    while (hash_table_status[index] == 1 && hash_table[index] != key) {
        index = (index + 1) % NUM_BUCKETS;
        i++;
        if (i == NUM_BUCKETS) {
            return false;
        }
    }
    return hash_table_status[index] == 1;
}

__global__ void initialize_hash_table() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_BUCKETS) {
        hash_table[idx] = 0;
        hash_table_status[idx] = 0;
    }
}

__global__ void insert_keys(int *keys, int num_keys) {
    extern __shared__ int shared_keys[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        shared_keys[threadIdx.x] = keys[idx];
        __syncthreads();

        // Insert keys using dynamic parallelism
        insert_key(shared_keys[threadIdx.x]);
    }
}

__global__ void search_keys(int *keys, int num_keys, int *results) {
    extern __shared__ int shared_keys[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        shared_keys[threadIdx.x] = keys[idx];
        __syncthreads();

        // Search keys using dynamic parallelism
        results[idx] = search_key(shared_keys[threadIdx.x]);
    }
}

int main() {
    int num_keys = 512;
    int *h_keys = new int[num_keys];
    int *h_results = new int[num_keys];
    int *d_keys, *d_results;

    // Initialize keys
    for (int i = 0; i < num_keys; i++) {
        h_keys[i] = rand() % MAX_KEY;
        h_results[i] = 0;
    }

    // Allocate device memory
    hipError_t err = hipMalloc(&d_keys, num_keys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device keys memory" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_results, num_keys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device results memory" << std::endl;
        return -1;
    }

    // Copy keys to device
    err = hipMemcpy(d_keys, h_keys, num_keys * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy keys to device" << std::endl;
        return -1;
    }

    // Initialize hash table
    initialize_hash_table<<<(NUM_BUCKETS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
    hipDeviceSynchronize();

    // Insert keys
    insert_keys<<<(num_keys + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_keys, num_keys);
    hipDeviceSynchronize();

    // Search keys
    search_keys<<<(num_keys + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_keys, num_keys, d_results);
    hipDeviceSynchronize();

    // Copy results back to host
    err = hipMemcpy(h_results, d_results, num_keys * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy results to host" << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_results);

    // Clean up
    delete[] h_keys;
    delete[] h_results;

    std::cout << "Hash table operations completed successfully" << std::endl;
    return 0;
}