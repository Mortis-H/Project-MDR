// skip_list_cuda.cu
#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 1024
#define MAX_LEVEL 16

// Node structure for skip list
struct Node {
    int value;
    int level;
    int next[MAX_LEVEL];
};

// Kernel to initialize the skip list nodes
__global__ void initSkipListKernel(Node *d_nodes, int *d_randomLevels, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_nodes[idx].value = idx; // Initialize with index value for simplicity
        d_nodes[idx].level = d_randomLevels[idx] % MAX_LEVEL; // Assign random level
        for (int i = 0; i < MAX_LEVEL; i++) {
            d_nodes[idx].next[i] = -1; // Initialize next pointers to -1 (null)
        }
    }
}

// Kernel to insert elements into the skip list
__global__ void insertSkipListKernel(Node *d_nodes, int n) {
    extern __shared__ int sharedNext[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Load the next pointers of the current node into shared memory
    for (int i = 0; i <= d_nodes[idx].level; i++) {
        sharedNext[i] = d_nodes[idx].next[i];
    }
    __syncthreads();

    // Insert logic here (simplified for demonstration)
    // This is a placeholder for the actual insertion logic
    // which would involve finding the correct position and updating pointers
    for (int i = 0; i <= d_nodes[idx].level; i++) {
        d_nodes[idx].next[i] = sharedNext[i] + 1; // Example update
    }
}

int main() {
    Node *d_nodes;
    int *d_randomLevels;
    Node h_nodes[N];
    int h_randomLevels[N];

    // Initialize random levels for demonstration
    for (int i = 0; i < N; i++) {
        h_randomLevels[i] = rand() % MAX_LEVEL;
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_nodes, N * sizeof(Node));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for nodes" << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_randomLevels, N * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for random levels" << std::endl;
        hipFree(d_nodes);
        return -1;
    }

    // Copy data to device
    err = hipMemcpy(d_randomLevels, h_randomLevels, N * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy random levels to device" << std::endl;
        hipFree(d_nodes);
        hipFree(d_randomLevels);
        return -1;
    }

    // Initialize skip list
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    initSkipListKernel<<<numBlocks, blockSize>>>(d_nodes, d_randomLevels, N);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel execution failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_randomLevels);
        return -1;
    }

    // Insert elements into skip list
    insertSkipListKernel<<<numBlocks, blockSize, (MAX_LEVEL + 1) * sizeof(int)>>>(d_nodes, N);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel execution failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_nodes);
        hipFree(d_randomLevels);
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_nodes, d_nodes, N * sizeof(Node), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy nodes back to host" << std::endl;
        hipFree(d_nodes);
        hipFree(d_randomLevels);
        return -1;
    }

    // Free device memory
    hipFree(d_nodes);
    hipFree(d_randomLevels);

    // Output some results for verification
    for (int i = 0; i < 10; i++) {
        std::cout << "Node " << i << ": Value = " << h_nodes[i].value << ", Level = " << h_nodes[i].level;
        for (int j = 0; j <= h_nodes[i].level; j++) {
            std::cout << ", Next[" << j << "] = " << h_nodes[i].next[j];
        }
        std::cout << std::endl;
    }

    return 0;
}