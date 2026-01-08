#include <iostream>
#include <hip/hip_runtime.h>


#define NUM_BUCKETS 16384
#define MAX_KEY 100000
#define MAX_THREADS 256

// Structure to represent a node in the linked list
struct Node {
    int key;
    int value;
    unsigned int next;
};

// Global memory for hash table buckets and nodes
__managed__ unsigned int buckets[NUM_BUCKETS];
__managed__ Node nodes[MAX_KEY];

// Kernel to initialize hash table buckets
__global__ void initBuckets() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_BUCKETS) {
        buckets[idx] = 0xFFFFFFFF; // 0xFFFFFFFF indicates empty bucket
    }
}

// Kernel to insert a key-value pair into the hash table
__global__ void insertKernel(int key, int value) {
    unsigned int idx = key % NUM_BUCKETS;
    unsigned int nodeIdx = key; // Using key as node index for simplicity

    // Create new node
    nodes[nodeIdx].key = key;
    nodes[nodeIdx].value = value;
    nodes[nodeIdx].next = buckets[idx];

    // Insert node at the beginning of the list
    buckets[idx] = nodeIdx;
}

// Kernel to search for a key in the hash table
__global__ void searchKernel(int key, int* result) {
    unsigned int idx = key % NUM_BUCKETS;
    unsigned int nodeIdx = buckets[idx];

    while (nodeIdx != 0xFFFFFFFF) {
        if (nodes[nodeIdx].key == key) {
            *result = nodes[nodeIdx].value;
            return;
        }
        nodeIdx = nodes[nodeIdx].next;
    }
    *result = -1; // Key not found
}

int main() {
    int numKeys = 1000;
    int* h_keys = new int[numKeys];
    int* h_values = new int[numKeys];
    int* d_result;

    // Initialize keys and values
    for (int i = 0; i < numKeys; i++) {
        h_keys[i] = i;
        h_values[i] = i * 2;
    }

    // Allocate device memory for result
    hipError_t err = hipMalloc(&d_result, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for result: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize buckets
    initBuckets<<<(NUM_BUCKETS + MAX_THREADS - 1) / MAX_THREADS, MAX_THREADS>>>();
    hipDeviceSynchronize();

    // Insert keys and values into the hash table
    for (int i = 0; i < numKeys; i++) {
        insertKernel<<<1, 1>>>(h_keys[i], h_values[i]);
        hipDeviceSynchronize();
    }

    // Search for a key
    int searchKey = 500;
    searchKernel<<<1, 1>>>(searchKey, d_result);
    hipDeviceSynchronize();

    // Copy result back to host
    int result;
    hipMemcpy(&result, d_result, sizeof(int), hipMemcpyDeviceToHost);

    std::cout << "Value for key " << searchKey << ": " << result << std::endl;

    // Free device memory
    hipFree(d_result);

    // Free host memory
    delete[] h_keys;
    delete[] h_values;

    return 0;
}