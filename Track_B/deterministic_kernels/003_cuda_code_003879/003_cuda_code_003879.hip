#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define constants
#define TABLE_SIZE 2048
#define NUM_GPUS 2
#define BLOCK_SIZE 256

// Kernel to initialize the hash table
__global__ void initHashTable(int* hashTable) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < TABLE_SIZE) {
        hashTable[idx] = -1; // Initialize with -1 to indicate empty
    }
}

// Kernel to insert elements into the hash table
__global__ void insertIntoHashTable(int* hashTable, int* keys, int* values, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        int key = keys[idx];
        int value = values[idx];
        int hashIndex = key % TABLE_SIZE;
        while (atomicCAS(&hashTable[hashIndex], -1, key) != -1) {
            hashIndex = (hashIndex + 1) % TABLE_SIZE; // Linear probing
        }
        hashTable[hashIndex + TABLE_SIZE] = value; // Store value
    }
}

// Kernel to search for elements in the hash table
__global__ void searchHashTable(int* hashTable, int* keys, int* results, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        int key = keys[idx];
        int hashIndex = key % TABLE_SIZE;
        while (hashTable[hashIndex] != key && hashTable[hashIndex] != -1) {
            hashIndex = (hashIndex + 1) % TABLE_SIZE; // Linear probing
        }
        results[idx] = (hashTable[hashIndex] == key) ? hashTable[hashIndex + TABLE_SIZE] : -1;
    }
}

// Function to check for CUDA errors
void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int numElements = TABLE_SIZE;
    int* h_keys = new int[numElements];
    int* h_values = new int[numElements];
    int* h_results = new int[numElements];

    // Initialize keys and values
    for (int i = 0; i < numElements; ++i) {
        h_keys[i] = i;
        h_values[i] = i * 10;
        h_results[i] = -1;
    }

    int* d_hashTables[NUM_GPUS];
    int* d_keys[NUM_GPUS];
    int* d_values[NUM_GPUS];
    int* d_results[NUM_GPUS];

    // Initialize CUDA devices
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);

        // Allocate memory on the device
        checkCudaError(hipMalloc(&d_hashTables[gpu], TABLE_SIZE * 2 * sizeof(int)), "Failed to allocate hash table memory");
        checkCudaError(hipMalloc(&d_keys[gpu], numElements * sizeof(int)), "Failed to allocate keys memory");
        checkCudaError(hipMalloc(&d_values[gpu], numElements * sizeof(int)), "Failed to allocate values memory");
        checkCudaError(hipMalloc(&d_results[gpu], numElements * sizeof(int)), "Failed to allocate results memory");

        // Copy data to the device
        checkCudaError(hipMemcpy(d_keys[gpu], h_keys, numElements * sizeof(int), hipMemcpyHostToDevice), "Failed to copy keys to device");
        checkCudaError(hipMemcpy(d_values[gpu], h_values, numElements * sizeof(int), hipMemcpyHostToDevice), "Failed to copy values to device");

        // Initialize hash table
        int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        initHashTable<<<numBlocks, BLOCK_SIZE>>>(d_hashTables[gpu]);
        checkCudaError(hipGetLastError(), "Failed to launch initHashTable kernel");
        checkCudaError(hipDeviceSynchronize(), "Failed to synchronize device");

        // Insert elements into hash table
        insertIntoHashTable<<<numBlocks, BLOCK_SIZE>>>(d_hashTables[gpu], d_keys[gpu], d_values[gpu], numElements);
        checkCudaError(hipGetLastError(), "Failed to launch insertIntoHashTable kernel");
        checkCudaError(hipDeviceSynchronize(), "Failed to synchronize device");

        // Search elements in hash table
        searchHashTable<<<numBlocks, BLOCK_SIZE>>>(d_hashTables[gpu], d_keys[gpu], d_results[gpu], numElements);
        checkCudaError(hipGetLastError(), "Failed to launch searchHashTable kernel");
        checkCudaError(hipDeviceSynchronize(), "Failed to synchronize device");

        // Copy results back to host
        checkCudaError(hipMemcpy(h_results, d_results[gpu], numElements * sizeof(int), hipMemcpyDeviceToHost), "Failed to copy results to host");
    }

    // Free device memory
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        hipFree(d_hashTables[gpu]);
        hipFree(d_keys[gpu]);
        hipFree(d_values[gpu]);
        hipFree(d_results[gpu]);
    }

    // Verify results
    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        if (h_results[i] != h_values[i]) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Hash table operations successful!" << std::endl;
    } else {
        std::cout << "Hash table operations failed!" << std::endl;
    }

    // Free host memory
    delete[] h_keys;
    delete[] h_values;
    delete[] h_results;

    return 0;
}