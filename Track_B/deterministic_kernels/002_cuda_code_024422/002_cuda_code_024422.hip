#include <iostream>
#include <hip/hip_runtime.h>


#define NUM_BUCKETS 64
#define MAX_THREADS 1024

__device__ int hash_table[NUM_BUCKETS];

__device__ int hash_function(int key) {
    return key % NUM_BUCKETS;
}

__device__ int quadratic_probing(int key, int i) {
    return (hash_function(key) + i * i) % NUM_BUCKETS;
}

__global__ void insert_kernel(int *keys, int num_keys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        int key = keys[idx];
        int i = 0;
        int bucket;
        do {
            bucket = quadratic_probing(key, i);
            i++;
        } while (atomicCAS(&hash_table[bucket], 0, key) != 0);
    }
}

__global__ void search_kernel(int *keys, int num_keys, int *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        int key = keys[idx];
        int i = 0;
        int bucket;
        bool found = false;
        do {
            bucket = quadratic_probing(key, i);
            if (hash_table[bucket] == key) {
                results[idx] = 1;
                found = true;
                break;
            }
            i++;
        } while (hash_table[bucket] != 0 && i < NUM_BUCKETS);
        if (!found) {
            results[idx] = 0;
        }
    }
}

int main() {
    int num_keys = 128;
    int *keys, *d_keys, *results, *d_results;

    // Allocate host memory
    keys = (int *)malloc(num_keys * sizeof(int));
    results = (int *)malloc(num_keys * sizeof(int));

    // Initialize keys
    for (int i = 0; i < num_keys; i++) {
        keys[i] = i * 3 + 1; // Arbitrary keys
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void **)&d_keys, num_keys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device keys memory" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_results, num_keys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device results memory" << std::endl;
        hipFree(d_keys);
        return -1;
    }

    // Copy keys to device
    err = hipMemcpy(d_keys, keys, num_keys * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy keys to device" << std::endl;
        hipFree(d_keys);
        hipFree(d_results);
        return -1;
    }

    // Initialize hash table to 0
    err = hipMemset(hash_table, 0, NUM_BUCKETS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize hash table" << std::endl;
        hipFree(d_keys);
        hipFree(d_results);
        return -1;
    }

    // Launch insert kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_keys + threadsPerBlock - 1) / threadsPerBlock;
    insert_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_keys, num_keys);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch insert kernel" << std::endl;
        hipFree(d_keys);
        hipFree(d_results);
        return -1;
    }

    // Launch search kernel
    search_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_keys, num_keys, d_results);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch search kernel" << std::endl;
        hipFree(d_keys);
        hipFree(d_results);
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(results, d_results, num_keys * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy results to host" << std::endl;
        hipFree(d_keys);
        hipFree(d_results);
        return -1;
    }

    // Print results
    for (int i = 0; i < num_keys; i++) {
        std::cout << "Key " << keys[i] << ": " << (results[i] ? "Found" : "Not Found") << std::endl;
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_results);

    // Free host memory
    free(keys);
    free(results);

    return 0;
}