#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_ELEMENTS 128
#define MAX_LEVEL 16

typedef struct Node {
    int key;
    int level;
    int next[MAX_LEVEL];
} Node;

__device__ Node skipList[NUM_ELEMENTS];
__device__ int listHead;

__device__ int randomLevel(hiprandState *localState) {
    int lvl = 1;
    while (hiprand_uniform(localState) < 0.5 && lvl < MAX_LEVEL) lvl++;
    return lvl;
}

__global__ void initializeSkipList() {
    extern __shared__ hiprandState localState[];
    int id = threadIdx.x;
    hiprand_init(1234, id, 0, &localState[id]);

    if (id == 0) {
        listHead = 0;
        skipList[0].key = INT_MIN;
        skipList[0].level = MAX_LEVEL;
        for (int i = 0; i < MAX_LEVEL; i++) {
            skipList[0].next[i] = NUM_ELEMENTS; // Use NUM_ELEMENTS to indicate NULL
        }
    }
}

__global__ void insertElement(int *elements, int numElements) {
    extern __shared__ hiprandState localState[];
    int id = threadIdx.x;
    hiprand_init(1234, id, 0, &localState[id]);

    if (id < numElements) {
        int key = elements[id];
        int level = randomLevel(&localState[id]);
        int update[MAX_LEVEL];

        int current = listHead;
        for (int i = MAX_LEVEL - 1; i >= 0; i--) {
            while (skipList[current].next[i] < NUM_ELEMENTS && skipList[skipList[current].next[i]].key < key) {
                current = skipList[current].next[i];
            }
            update[i] = current;
        }

        int newNodeIndex = id + 1; // Assuming elements are inserted in order of threads
        skipList[newNodeIndex].key = key;
        skipList[newNodeIndex].level = level;

        for (int i = 0; i < level; i++) {
            skipList[newNodeIndex].next[i] = skipList[update[i]].next[i];
            skipList[update[i]].next[i] = newNodeIndex;
        }
    }
}

int main() {
    int *d_elements;
    int h_elements[NUM_ELEMENTS - 1];

    // Initialize host elements with random values
    for (int i = 0; i < NUM_ELEMENTS - 1; i++) {
        h_elements[i] = i + 1; // Simple ascending order for demonstration
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void **)&d_elements, (NUM_ELEMENTS - 1) * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy elements to device
    err = hipMemcpy(d_elements, h_elements, (NUM_ELEMENTS - 1) * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying data to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_elements);
        return 1;
    }

    // Initialize skip list
    initializeSkipList<<<1, 1, sizeof(hiprandState)>>>();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing skip list: " << hipGetErrorString(err) << std::endl;
        hipFree(d_elements);
        return 1;
    }

    // Insert elements into skip list
    insertElement<<<1, NUM_ELEMENTS - 1, sizeof(hiprandState)>>>(d_elements, NUM_ELEMENTS - 1);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error inserting elements into skip list: " << hipGetErrorString(err) << std::endl;
        hipFree(d_elements);
        return 1;
    }

    // Free device memory
    hipFree(d_elements);

    std::cout << "Skip list initialization and insertion complete." << std::endl;

    return 0;
}