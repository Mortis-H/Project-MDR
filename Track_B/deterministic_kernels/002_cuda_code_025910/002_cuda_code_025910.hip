#include <iostream>
#include <hip/hip_runtime.h>

// Define the maximum level of the skip list
#define MAX_LEVEL 10

// Define the node structure for the skip list
struct Node {
    int value;
    Node* forward[MAX_LEVEL];
};

// Function to create a new node
__device__ Node* createNode(int value, int level) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->value = value;
    for (int i = 0; i <= level; i++) {
        node->forward[i] = nullptr;
    }
    return node;
}

// Kernel to insert elements into the skip list
__global__ void insertSkipList(Node* header, int* values, int numValues) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numValues) return;

    Node* current = header;
    int value = values[idx];

    // Create a new node
    Node* newNode = createNode(value, MAX_LEVEL);

    // Find the position to insert the new node
    for (int i = MAX_LEVEL; i >= 0; i--) {
        while (current->forward[i] != nullptr && current->forward[i]->value < value) {
            current = current->forward[i];
        }
        // Link the new node
        newNode->forward[i] = current->forward[i];
        current->forward[i] = newNode;
    }
}

// Kernel to search for an element in the skip list
__global__ void searchSkipList(Node* header, int* searchValues, bool* results, int numSearchValues) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numSearchValues) return;

    Node* current = header;
    int value = searchValues[idx];
    bool found = false;

    // Traverse the skip list to find the value
    for (int i = MAX_LEVEL; i >= 0; i--) {
        while (current->forward[i] != nullptr && current->forward[i]->value < value) {
            current = current->forward[i];
        }
    }
    // Check if the value is found
    if (current->forward[0] != nullptr && current->forward[0]->value == value) {
        found = true;
    }
    results[idx] = found;
}

int main() {
    // Number of elements to insert and search
    const int numElements = 1024;
    const int numSearchElements = 100;

    // Allocate memory for values to insert and search
    int* h_values = (int*)malloc(numElements * sizeof(int));
    int* h_searchValues = (int*)malloc(numSearchElements * sizeof(int));
    bool* h_results = (bool*)malloc(numSearchElements * sizeof(bool));

    // Initialize values
    for (int i = 0; i < numElements; i++) {
        h_values[i] = i * 2; // Even numbers
    }
    for (int i = 0; i < numSearchElements; i++) {
        h_searchValues[i] = i * 2; // Search for even numbers
    }

    // Allocate Unified Memory for the skip list header
    Node* d_header;
    hipMallocManaged(&d_header, sizeof(Node));
    d_header->value = -1; // Sentinel value
    for (int i = 0; i <= MAX_LEVEL; i++) {
        d_header->forward[i] = nullptr;
    }

    // Allocate Unified Memory for values to insert and search
    int* d_values;
    int* d_searchValues;
    bool* d_results;
    hipMallocManaged(&d_values, numElements * sizeof(int));
    hipMallocManaged(&d_searchValues, numSearchElements * sizeof(int));
    hipMallocManaged(&d_results, numSearchElements * sizeof(bool));

    // Copy data to device
    hipMemcpy(d_values, h_values, numElements * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_searchValues, h_searchValues, numSearchElements * sizeof(int), hipMemcpyHostToDevice);

    // Define grid and block sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Insert elements into the skip list
    insertSkipList<<<blocksPerGrid, threadsPerBlock>>>(d_header, d_values, numElements);
    hipDeviceSynchronize();

    // Search for elements in the skip list
    blocksPerGrid = (numSearchElements + threadsPerBlock - 1) / threadsPerBlock;
    searchSkipList<<<blocksPerGrid, threadsPerBlock>>>(d_header, d_searchValues, d_results, numSearchElements);
    hipDeviceSynchronize();

    // Copy results back to host
    hipMemcpy(h_results, d_results, numSearchElements * sizeof(bool), hipMemcpyDeviceToHost);

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_header);
    hipFree(d_values);
    hipFree(d_searchValues);
    hipFree(d_results);

    // Free host memory
    free(h_values);
    free(h_searchValues);
    free(h_results);

    std::cout << "Skip list operations completed successfully." << std::endl;
    return 0;
}