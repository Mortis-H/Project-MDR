#include <iostream>
#include <hip/hip_runtime.h>

#define TABLE_SIZE 4096
#define HASH_MASK (TABLE_SIZE - 1)

// Kernel to insert keys into the hash table using quadratic probing
__global__ void insertKeys(int* keys, int* table, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numKeys) return;

    int key = keys[idx];
    int hash = key & HASH_MASK;
    int i = 0;

    // Quadratic probing to find an empty slot
    while (true) {
        int slot = (hash + i * i) & HASH_MASK;
        int expected = 0;
        // Use atomicCAS to avoid race conditions
        if (atomicCAS(&table[slot], expected, key) == expected) {
            break;
        }
        i++;
    }
}

// Kernel to search for keys in the hash table using quadratic probing
__global__ void searchKeys(int* keys, int* table, int* results, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numKeys) return;

    int key = keys[idx];
    int hash = key & HASH_MASK;
    int i = 0;
    int found = -1;

    // Quadratic probing to find the key
    while (i < TABLE_SIZE) {
        int slot = (hash + i * i) & HASH_MASK;
        if (table[slot] == key) {
            found = 1;
            break;
        } else if (table[slot] == 0) {
            break;
        }
        i++;
    }

    results[idx] = found;
}

int main() {
    int numKeys = 1024; // Number of keys to insert and search
    int* h_keys = new int[numKeys];
    int* h_table = new int[TABLE_SIZE];
    int* h_results = new int[numKeys];

    // Initialize keys and table
    for (int i = 0; i < numKeys; i++) {
        h_keys[i] = i * 2 + 1; // Odd numbers to avoid collisions for simplicity
    }
    for (int i = 0; i < TABLE_SIZE; i++) {
        h_table[i] = 0;
    }

    // Allocate device memory
    int* d_keys;
    int* d_table;
    int* d_results;
    hipError_t err;

    err = hipMalloc((void**)&d_keys, numKeys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_table, TABLE_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_table: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_results, numKeys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_results: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy data to device
    err = hipMemcpy(d_keys, h_keys, numKeys * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying h_keys to d_keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMemcpy(d_table, h_table, TABLE_SIZE * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying h_table to d_table: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch insert kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numKeys + threadsPerBlock - 1) / threadsPerBlock;
    insertKeys<<<blocksPerGrid, threadsPerBlock>>>(d_keys, d_table, numKeys);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching insertKeys kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch search kernel
    searchKeys<<<blocksPerGrid, threadsPerBlock>>>(d_keys, d_table, d_results, numKeys);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching searchKeys kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_results, d_results, numKeys * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying d_results to h_results: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_table);
    hipFree(d_results);

    // Free host memory
    delete[] h_keys;
    delete[] h_table;
    delete[] h_results;

    std::cout << "Hash table operations completed successfully." << std::endl;
    return 0;
}