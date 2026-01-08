#include <iostream>
#include <hip/hip_runtime.h>
#include <vector>

#define BUCKET_COUNT 4096
#define MAX_ELEMENTS 1024

// Structure for a node in the linked list
struct Node {
    int key;
    int value;
    int next;
};

// Hash function
__device__ int hash(int key) {
    return key % BUCKET_COUNT;
}

// Kernel to insert elements into the hash table
__global__ void insertKernel(Node* nodes, int* head, int* count, int* keys, int* values, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        int key = keys[idx];
        int value = values[idx];
        int bucket = hash(key);

        // Allocate a new node
        int newNodeIdx = atomicAdd(count, 1);
        if (newNodeIdx >= MAX_ELEMENTS) {
            return; // No more space
        }

        nodes[newNodeIdx].key = key;
        nodes[newNodeIdx].value = value;
        nodes[newNodeIdx].next = -1;

        // Insert the new node at the head of the bucket
        int oldHead = atomicExch(&head[bucket], newNodeIdx);
        if (oldHead != -1) {
            nodes[newNodeIdx].next = oldHead;
        }
    }
}

// Kernel to search for elements in the hash table
__global__ void searchKernel(Node* nodes, int* head, int* keys, int* results, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        int key = keys[idx];
        int bucket = hash(key);
        int nodeIdx = head[bucket];

        // Traverse the linked list in the bucket
        while (nodeIdx != -1) {
            if (nodes[nodeIdx].key == key) {
                results[idx] = nodes[nodeIdx].value;
                return;
            }
            nodeIdx = nodes[nodeIdx].next;
        }
        results[idx] = -1; // Key not found
    }
}

int main() {
    // Host data
    std::vector<int> h_keys = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> h_values = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    std::vector<int> h_search_keys = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    std::vector<int> h_results(h_search_keys.size(), -1);

    // Device data
    Node* d_nodes;
    int* d_head;
    int* d_count;
    int* d_keys;
    int* d_values;
    int* d_search_keys;
    int* d_results;

    // Allocate memory on device
    hipMalloc(&d_nodes, MAX_ELEMENTS * sizeof(Node));
    hipMalloc(&d_head, BUCKET_COUNT * sizeof(int));
    hipMalloc(&d_count, sizeof(int));
    hipMalloc(&d_keys, h_keys.size() * sizeof(int));
    hipMalloc(&d_values, h_values.size() * sizeof(int));
    hipMalloc(&d_search_keys, h_search_keys.size() * sizeof(int));
    hipMalloc(&d_results, h_results.size() * sizeof(int));

    // Initialize device data
    hipMemset(d_head, -1, BUCKET_COUNT * sizeof(int));
    hipMemset(d_count, 0, sizeof(int));
    hipMemcpy(d_keys, h_keys.data(), h_keys.size() * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_values, h_values.data(), h_values.size() * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_search_keys, h_search_keys.data(), h_search_keys.size() * sizeof(int), hipMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(256);
    dim3 numBlocks((h_keys.size() + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Insert elements into the hash table
    insertKernel<<<numBlocks, threadsPerBlock>>>(d_nodes, d_head, d_count, d_keys, d_values, h_keys.size());
    hipDeviceSynchronize();

    // Search for elements in the hash table
    searchKernel<<<numBlocks, threadsPerBlock>>>(d_nodes, d_head, d_search_keys, d_results, h_search_keys.size());
    hipDeviceSynchronize();

    // Copy results back to host
    hipMemcpy(h_results.data(), d_results, h_results.size() * sizeof(int), hipMemcpyDeviceToHost);

    // Print results
    for (size_t i = 0; i < h_results.size(); ++i) {
        std::cout << "Key: " << h_search_keys[i] << ", Value: " << h_results[i] << std::endl;
    }

    // Free device memory
    hipFree(d_nodes);
    hipFree(d_head);
    hipFree(d_count);
    hipFree(d_keys);
    hipFree(d_values);
    hipFree(d_search_keys);
    hipFree(d_results);

    return 0;
}