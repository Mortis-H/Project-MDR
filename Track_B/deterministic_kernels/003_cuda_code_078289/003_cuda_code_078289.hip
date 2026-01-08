#include <iostream>
#include <hip/hip_runtime.h>

// Define the maximum number of levels in the skip list
#define MAX_LEVEL 16

// Node structure for the skip list
struct SkipListNode {
    int value;
    SkipListNode* forward[MAX_LEVEL];
};

// Device function to initialize a node
__device__ void initNode(SkipListNode* node, int value) {
    node->value = value;
    for (int i = 0; i < MAX_LEVEL; ++i) {
        node->forward[i] = nullptr;
    }
}

// Kernel to initialize the skip list
__global__ void initializeSkipList(SkipListNode* nodes, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        initNode(&nodes[idx], idx); // Initialize node with its index as value
    }
}

// Kernel to insert a value into the skip list
__global__ void insertIntoSkipList(SkipListNode* nodes, int value, int numElements) {
    extern __shared__ SkipListNode* update[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        // Initialize update array
        for (int i = 0; i < MAX_LEVEL; ++i) {
            update[i] = &nodes[numElements - 1]; // Point to the last node initially
        }

        // Traverse the list from the highest level to the lowest
        for (int level = MAX_LEVEL - 1; level >= 0; --level) {
            while (update[level]->forward[level] != nullptr && update[level]->forward[level]->value < value) {
                update[level] = update[level]->forward[level];
            }
        }

        // Insert the new node at the appropriate level
        SkipListNode* newNode = &nodes[value];
        for (int level = 0; level < MAX_LEVEL; ++level) {
            newNode->forward[level] = update[level]->forward[level];
            update[level]->forward[level] = newNode;
        }
    }
}

// Kernel to search for a value in the skip list
__global__ void searchInSkipList(SkipListNode* nodes, int value, int* result, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        SkipListNode* current = &nodes[numElements - 1]; // Start from the last node

        // Traverse the list from the highest level to the lowest
        for (int level = MAX_LEVEL - 1; level >= 0; --level) {
            while (current->forward[level] != nullptr && current->forward[level]->value < value) {
                current = current->forward[level];
            }
        }

        // Move to the next level
        current = current->forward[0];

        // Check if the value is found
        if (current != nullptr && current->value == value) {
            *result = 1; // Value found
        } else {
            *result = 0; // Value not found
        }
    }
}

int main() {
    const int numElements = 1024;
    const int blockSize = 256;
    const int numBlocks = (numElements + blockSize - 1) / blockSize;

    SkipListNode* d_nodes;
    int* d_result;

    // Allocate unified memory for nodes and result
    hipError_t err = hipMallocManaged(&d_nodes, numElements * sizeof(SkipListNode));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate unified memory for nodes" << std::endl;
        return -1;
    }

    err = hipMallocManaged(&d_result, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate unified memory for result" << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Initialize the skip list
    initializeSkipList<<<numBlocks, blockSize>>>(d_nodes, numElements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize skip list: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_result);
        return -1;
    }

    hipDeviceSynchronize();

    // Insert a value into the skip list
    int valueToInsert = 512;
    insertIntoSkipList<<<1, 1, MAX_LEVEL * sizeof(SkipListNode*)>>>(d_nodes, valueToInsert, numElements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to insert into skip list: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_result);
        return -1;
    }

    hipDeviceSynchronize();

    // Search for a value in the skip list
    int valueToSearch = 512;
    *d_result = -1; // Initialize result to -1
    searchInSkipList<<<1, 1>>>(d_nodes, valueToSearch, d_result, numElements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to search in skip list: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_result);
        return -1;
    }

    hipDeviceSynchronize();

    // Output the result of the search
    if (*d_result == 1) {
        std::cout << "Value " << valueToSearch << " found in the skip list." << std::endl;
    } else {
        std::cout << "Value " << valueToSearch << " not found in the skip list." << std::endl;
    }

    // Free the allocated memory
    hipFree(d_nodes);
    hipFree(d_result);

    return 0;
}