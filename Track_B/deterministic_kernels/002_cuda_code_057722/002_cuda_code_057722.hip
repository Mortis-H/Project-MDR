#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define the maximum level of the skip list
#define MAX_LEVEL 16

// Define the structure for a node in the skip list
struct Node {
    int key;
    Node* forward[MAX_LEVEL];
};

// Kernel to initialize random number generators
__global__ void initCurandStates(hiprandState *states, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    hiprand_init(seed, id, 0, &states[id]);
}

// Kernel to perform skip list insertions
__global__ void insertSkipList(Node* nodes, hiprandState *states, int numElements) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= numElements) return;

    hiprandState localState = states[id];
    Node* update[MAX_LEVEL];
    Node* x = nodes;

    // Find the position to insert the new node
    for (int i = MAX_LEVEL - 1; i >= 0; i--) {
        while (x->forward[i] != nullptr && x->forward[i]->key < id) {
            x = x->forward[i];
        }
        update[i] = x;
    }

    // Generate a random level for the new node
    int lvl = 1;
    while (lvl < MAX_LEVEL && hiprand_uniform(&localState) < 0.5) lvl++;

    // Allocate a new node and set its key
    Node* newNode = &nodes[id];
    newNode->key = id;

    // Insert the new node into the skip list
    for (int i = 0; i < lvl; i++) {
        newNode->forward[i] = update[i]->forward[i];
        update[i]->forward[i] = newNode;
    }

    states[id] = localState;
}

int main() {
    const int numElements = 16384;
    const int numThreads = 256;
    const int numBlocks = (numElements + numThreads - 1) / numThreads;

    // Allocate memory for nodes and random states
    Node* d_nodes;
    hiprandState *d_states;
    hipError_t err;

    err = hipMallocManaged(&d_nodes, numElements * sizeof(Node));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for nodes: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMallocManaged(&d_states, numElements * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for random states: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Initialize random number generators
    initCurandStates<<<numBlocks, numThreads>>>(d_states, time(NULL));
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize random number generators: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_states);
        return -1;
    }

    // Insert elements into the skip list
    insertSkipList<<<numBlocks, numThreads>>>(d_nodes, d_states, numElements);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to insert elements into the skip list: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_states);
        return -1;
    }

    // Free allocated memory
    hipFree(d_nodes);
    hipFree(d_states);

    std::cout << "Skip list operations completed successfully." << std::endl;
    return 0;
}