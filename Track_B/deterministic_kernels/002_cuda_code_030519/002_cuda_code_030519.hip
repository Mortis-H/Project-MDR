#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define BUCKET_SIZE 8
#define TABLE_SIZE (128 * BUCKET_SIZE)
#define WARP_SIZE 32

struct HashTableEntry {
    int key;
    int value;
    bool occupied;
};

__device__ int hash(int key) {
    return key % 128;
}

__device__ int quadraticProbe(int i) {
    return i * i;
}

__global__ void insertKernel(HashTableEntry *table, int *keys, int *values, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    int key = keys[idx];
    int value = values[idx];
    int bucketIdx = hash(key);
    int i = 0;

    while (i < BUCKET_SIZE) {
        int slot = (bucketIdx + quadraticProbe(i)) % TABLE_SIZE;
        int warpLane = threadIdx.x % WARP_SIZE;
        int warpOffset = warpLane * BUCKET_SIZE;
        int slotToCheck = slot + warpOffset;

        // Ensure each lane in the warp checks a different slot
        if (slotToCheck < TABLE_SIZE) {
            HashTableEntry entry = table[slotToCheck];
            if (!entry.occupied) {
                // Use atomicCAS to ensure only one thread can occupy the slot
                int *address_as_int = (int *)&table[slotToCheck].occupied;
                int expected = 0;
                int compare = atomicCAS(address_as_int, expected, 1);
                if (compare == expected) {
                    table[slotToCheck].key = key;
                    table[slotToCheck].value = value;
                    return;
                }
            }
        }
        i++;
    }
}

__global__ void lookupKernel(HashTableEntry *table, int *keys, int *results, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    int key = keys[idx];
    int bucketIdx = hash(key);
    int i = 0;

    while (i < BUCKET_SIZE) {
        int slot = (bucketIdx + quadraticProbe(i)) % TABLE_SIZE;
        int warpLane = threadIdx.x % WARP_SIZE;
        int warpOffset = warpLane * BUCKET_SIZE;
        int slotToCheck = slot + warpOffset;

        if (slotToCheck < TABLE_SIZE) {
            HashTableEntry entry = table[slotToCheck];
            if (entry.occupied && entry.key == key) {
                results[idx] = entry.value;
                return;
            }
        }
        i++;
    }
    results[idx] = -1; // Key not found
}

int main() {
    int numElements = 256;
    int *h_keys, *h_values, *h_results;
    int *d_keys, *d_values, *d_results;
    HashTableEntry *d_table;

    // Allocate host memory
    h_keys = (int *)malloc(numElements * sizeof(int));
    h_values = (int *)malloc(numElements * sizeof(int));
    h_results = (int *)malloc(numElements * sizeof(int));

    // Initialize host arrays
    for (int i = 0; i < numElements; i++) {
        h_keys[i] = i;
        h_values[i] = i * 10;
        h_results[i] = -1;
    }

    // Allocate unified memory
    hipMallocManaged(&d_table, TABLE_SIZE * sizeof(HashTableEntry));
    hipMallocManaged(&d_keys, numElements * sizeof(int));
    hipMallocManaged(&d_values, numElements * sizeof(int));
    hipMallocManaged(&d_results, numElements * sizeof(int));

    // Copy data to device
    hipMemcpy(d_keys, h_keys, numElements * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_values, h_values, numElements * sizeof(int), hipMemcpyHostToDevice);

    // Launch insert kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    insertKernel<<<blocksPerGrid, threadsPerBlock>>>(d_table, d_keys, d_values, numElements);
    hipDeviceSynchronize();

    // Launch lookup kernel
    lookupKernel<<<blocksPerGrid, threadsPerBlock>>>(d_table, d_keys, d_results, numElements);
    hipDeviceSynchronize();

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    hipMemcpy(h_results, d_results, numElements * sizeof(int), hipMemcpyDeviceToHost);

    // Verify results
    bool success = true;
    for (int i = 0; i < numElements; i++) {
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

    // Free memory
    hipFree(d_table);
    hipFree(d_keys);
    hipFree(d_values);
    hipFree(d_results);
    free(h_keys);
    free(h_values);
    free(h_results);

    return 0;
}