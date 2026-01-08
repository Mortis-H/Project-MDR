#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_ELEMENTS 4096
#define WARP_SIZE 32

// Node structure for the skip list
struct SkipListNode {
    int value;
    int level;
    int next[16]; // Assuming max level is 16
};

// Kernel to initialize skip list nodes
__global__ void initSkipListNodes(SkipListNode *nodes, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        nodes[idx].value = idx; // Initialize with index as value
        nodes[idx].level = 0;    // Base level
        for (int i = 0; i < 16; ++i) {
            nodes[idx].next[i] = -1; // -1 indicates no next node
        }
    }
}

// Kernel to perform skip list search using persistent threads
__global__ void skipListSearch(SkipListNode *nodes, int numElements, int *searchValues, int *results) {
    extern __shared__ int sharedResults[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;

    if (idx < numElements) {
        int searchValue = searchValues[idx];
        int current = 0;
        int level = 15; // Start from the highest level

        // Traverse the skip list
        while (level >= 0) {
            int next = nodes[current].next[level];
            if (next != -1 && nodes[next].value <= searchValue) {
                current = next;
            } else {
                level--;
            }
        }

        // Store the result in shared memory for coalesced write
        sharedResults[threadIdx.x] = nodes[current].value == searchValue ? current : -1;
        __syncthreads();

        // Write results back to global memory
        if (laneId == 0) {
            for (int i = 0; i < WARP_SIZE; ++i) {
                int globalIdx = idx + i;
                if (globalIdx < numElements) {
                    results[globalIdx] = sharedResults[i];
                }
            }
        }
    }
}

int main() {
    SkipListNode *d_nodes;
    int *d_searchValues, *d_results;

    // Allocate memory on device
    hipMalloc(&d_nodes, NUM_ELEMENTS * sizeof(SkipListNode));
    hipMalloc(&d_searchValues, NUM_ELEMENTS * sizeof(int));
    hipMalloc(&d_results, NUM_ELEMENTS * sizeof(int));

    // Initialize search values
    int *h_searchValues = new int[NUM_ELEMENTS];
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        h_searchValues[i] = i; // Search for each element
    }
    hipMemcpy(d_searchValues, h_searchValues, NUM_ELEMENTS * sizeof(int), hipMemcpyHostToDevice);

    // Initialize nodes on device
    int blockSize = 256;
    int numBlocks = (NUM_ELEMENTS + blockSize - 1) / blockSize;
    initSkipListNodes<<<numBlocks, blockSize>>>(d_nodes, NUM_ELEMENTS);
    hipDeviceSynchronize();

    // Perform skip list search
    skipListSearch<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_nodes, NUM_ELEMENTS, d_searchValues, d_results);
    hipDeviceSynchronize();

    // Copy results back to host
    int *h_results = new int[NUM_ELEMENTS];
    hipMemcpy(h_results, d_results, NUM_ELEMENTS * sizeof(int), hipMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        if (h_results[i] != i) {
            std::cerr << "Mismatch at index " << i << ": expected " << i << ", got " << h_results[i] << std::endl;
            return -1;
        }
    }

    // Free memory
    delete[] h_searchValues;
    delete[] h_results;
    hipFree(d_nodes);
    hipFree(d_searchValues);
    hipFree(d_results);

    std::cout << "Skip list search completed successfully." << std::endl;
    return 0;
}