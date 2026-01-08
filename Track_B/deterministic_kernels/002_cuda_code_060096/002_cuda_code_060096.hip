// hash_table.cu
// CUDA implementation of a hash table with 8192 elements, optimized for register usage on Volta architecture

#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define TABLE_SIZE 8192
#define WARP_SIZE 32

// Hash function for simplicity, real applications might use a more complex function
__device__ unsigned int hash(unsigned int key) {
    key = ~key + (key << 15);
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057;
    key = key ^ (key >> 16);
    return key % TABLE_SIZE;
}

// Structure to hold hash table entry
struct HashEntry {
    unsigned int key;
    unsigned int value;
};

// Kernel to insert elements into the hash table
__global__ void insertKernel(HashEntry *table, unsigned int *keys, unsigned int *values, int numElements) {
    extern __shared__ HashEntry sharedTable[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Load data into shared memory
    if (tid < TABLE_SIZE) {
        sharedTable[tid].key = 0;
        sharedTable[tid].value = 0;
    }
    __syncthreads();

    // Insert elements into the hash table
    if (tid < numElements) {
        unsigned int h = hash(keys[tid]);
        while (true) {
            unsigned int oldKey = atomicCAS(&sharedTable[h].key, 0, keys[tid]);
            if (oldKey == 0 || oldKey == keys[tid]) {
                sharedTable[h].value = values[tid];
                break;
            }
            h = (h + 1) % TABLE_SIZE;
        }
    }
    __syncthreads();

    // Write back to global memory
    if (tid < TABLE_SIZE) {
        table[tid] = sharedTable[tid];
    }
}

// Kernel to lookup elements in the hash table
__global__ void lookupKernel(HashEntry *table, unsigned int *keys, unsigned int *results, int numElements) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < numElements) {
        unsigned int h = hash(keys[tid]);
        while (true) {
            if (table[h].key == keys[tid]) {
                results[tid] = table[h].value;
                break;
            } else if (table[h].key == 0) {
                results[tid] = 0; // Key not found
                break;
            }
            h = (h + 1) % TABLE_SIZE;
        }
    }
}

int main() {
    const int numElements = 8192;
    HashEntry *d_table;
    unsigned int *d_keys, *d_values, *d_results;

    // Allocate memory on the device
    hipError_t err = hipMalloc(&d_table, TABLE_SIZE * sizeof(HashEntry));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for table" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_keys, numElements * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for keys" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_values, numElements * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for values" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_results, numElements * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for results" << std::endl;
        return -1;
    }

    // Initialize keys and values on the host
    unsigned int *h_keys = new unsigned int[numElements];
    unsigned int *h_values = new unsigned int[numElements];
    for (int i = 0; i < numElements; ++i) {
        h_keys[i] = i; // Simple keys for demonstration
        h_values[i] = i * 2; // Simple values for demonstration
    }

    // Copy keys and values to the device
    err = hipMemcpy(d_keys, h_keys, numElements * sizeof(unsigned int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy keys to device" << std::endl;
        return -1;
    }
    err = hipMemcpy(d_values, h_values, numElements * sizeof(unsigned int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy values to device" << std::endl;
        return -1;
    }

    // Launch the insert kernel
    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    insertKernel<<<numBlocks, blockSize, TABLE_SIZE * sizeof(HashEntry)>>>(d_table, d_keys, d_values, numElements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch insert kernel" << std::endl;
        return -1;
    }
    hipDeviceSynchronize();

    // Launch the lookup kernel
    lookupKernel<<<numBlocks, blockSize>>>(d_table, d_keys, d_results, numElements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch lookup kernel" << std::endl;
        return -1;
    }
    hipDeviceSynchronize();

    // Copy results back to the host
    unsigned int *h_results = new unsigned int[numElements];
    err = hipMemcpy(h_results, d_results, numElements * sizeof(unsigned int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy results from device" << std::endl;
        return -1;
    }

    // Verify results
    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        if (h_results[i] != h_values[i]) {
            std::cerr << "Mismatch at index " << i << ": expected " << h_values[i] << ", got " << h_results[i] << std::endl;
            success = false;
        }
    }
    if (success) {
        std::cout << "All lookups successful" << std::endl;
    }

    // Free memory
    delete[] h_keys;
    delete[] h_values;
    delete[] h_results;
    hipFree(d_table);
    hipFree(d_keys);
    hipFree(d_values);
    hipFree(d_results);

    return 0;
}