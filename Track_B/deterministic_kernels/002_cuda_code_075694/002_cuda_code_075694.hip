#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define the maximum number of elements per bucket
#define MAX_ELEMENTS_PER_BUCKET 10

// Define the hash table size
#define HASH_TABLE_SIZE 1024

// Define a structure for the hash table node
struct HashNode {
    int key;
    int value;
    int next;
};

// Kernel to insert elements into the hash table
__global__ void insertKernel(HashNode *hashTable, int *bucketHeads, int *keys, int *values, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        int key = keys[idx];
        int value = values[idx];
        int hashIndex = key % HASH_TABLE_SIZE;

        // Insert at the head of the list
        int oldHead = atomicExch(&bucketHeads[hashIndex], idx);
        hashTable[idx].key = key;
        hashTable[idx].value = value;
        hashTable[idx].next = oldHead;
    }
}

// Kernel to search for a key in the hash table
__global__ void searchKernel(HashNode *hashTable, int *bucketHeads, int searchKey, int *result) {
    int hashIndex = searchKey % HASH_TABLE_SIZE;
    int idx = bucketHeads[hashIndex];

    // Traverse the list
    while (idx != -1) {
        if (hashTable[idx].key == searchKey) {
            *result = hashTable[idx].value;
            return;
        }
        idx = hashTable[idx].next;
    }
    *result = -1; // Key not found
}

// Main function to demonstrate the usage of the hash table
int main() {
    int numElements = 2048; // Number of elements to insert

    // Allocate memory for the hash table on the host
    HashNode *h_hashTable = (HashNode *)malloc(numElements * sizeof(HashNode));
    int *h_bucketHeads = (int *)malloc(HASH_TABLE_SIZE * sizeof(int));
    int *h_keys = (int *)malloc(numElements * sizeof(int));
    int *h_values = (int *)malloc(numElements * sizeof(int));

    // Initialize the bucket heads to -1 (indicating empty)
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        h_bucketHeads[i] = -1;
    }

    // Initialize keys and values
    for (int i = 0; i < numElements; i++) {
        h_keys[i] = i; // Simple key initialization
        h_values[i] = i * 2; // Simple value initialization
    }

    // Allocate memory for the hash table on the device
    HashNode *d_hashTable;
    int *d_bucketHeads;
    int *d_keys;
    int *d_values;
    int *d_result;

    hipError_t err;

    err = hipMalloc((void **)&d_hashTable, numElements * sizeof(HashNode));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_hashTable: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void **)&d_bucketHeads, HASH_TABLE_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_bucketHeads: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void **)&d_keys, numElements * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void **)&d_values, numElements * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_values: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void **)&d_result, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_result: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_hashTable, h_hashTable, numElements * sizeof(HashNode), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying h_hashTable to d_hashTable: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMemcpy(d_bucketHeads, h_bucketHeads, HASH_TABLE_SIZE * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying h_bucketHeads to d_bucketHeads: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMemcpy(d_keys, h_keys, numElements * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying h_keys to d_keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMemcpy(d_values, h_values, numElements * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying h_values to d_values: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Set up the execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the insert kernel
    insertKernel<<<blocksPerGrid, threadsPerBlock>>>(d_hashTable, d_bucketHeads, d_keys, d_values, numElements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching insertKernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Synchronize to ensure all inserts are complete
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Error synchronizing device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Search for a key
    int searchKey = 1023;
    int h_result = -1;

    // Launch the search kernel
    searchKernel<<<1, 1>>>(d_hashTable, d_bucketHeads, searchKey, d_result);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching searchKernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy the result back to the host
    err = hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying d_result to h_result: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Output the result
    std::cout << "Value for key " << searchKey << " is " << h_result << std::endl;

    // Free device memory
    hipFree(d_hashTable);
    hipFree(d_bucketHeads);
    hipFree(d_keys);
    hipFree(d_values);
    hipFree(d_result);

    // Free host memory
    free(h_hashTable);
    free(h_bucketHeads);
    free(h_keys);
    free(h_values);

    return 0;
}