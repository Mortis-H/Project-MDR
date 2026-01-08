#include <iostream>
#include <hip/hip_runtime.h>

#define NUM_ELEMENTS 4096
#define BLOCK_SIZE 256

// Structure to represent a node in the octree
struct OctreeNode {
    float data;
    int children[8]; // Indices of the children nodes
};

// Kernel to initialize the octree nodes
__global__ void initializeOctree(OctreeNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        // Initialize node data and children indices
        nodes[idx].data = static_cast<float>(idx);
        for (int i = 0; i < 8; ++i) {
            nodes[idx].children[i] = -1; // -1 indicates no child
        }
    }
}

// Kernel to demonstrate some operations on the octree
__global__ void processOctree(OctreeNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        // Example operation: increment data by 1
        nodes[idx].data += 1.0f;
    }
}

// Error checking macro
#define cudaCheckError(err) (cudaCheckErrorImpl(err, __FILE__, __LINE__))

inline void cudaCheckErrorImpl(hipError_t err, const char* file, int line) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Allocate memory for the octree nodes on the host
    OctreeNode* h_nodes = new OctreeNode[NUM_ELEMENTS];

    // Allocate memory for the octree nodes on the device
    OctreeNode* d_nodes;
    cudaCheckError(hipMalloc(&d_nodes, NUM_ELEMENTS * sizeof(OctreeNode)));

    // Launch the kernel to initialize the octree nodes
    int numBlocks = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializeOctree<<<numBlocks, BLOCK_SIZE>>>(d_nodes);
    cudaCheckError(hipGetLastError());
    cudaCheckError(hipDeviceSynchronize());

    // Launch the kernel to process the octree nodes
    processOctree<<<numBlocks, BLOCK_SIZE>>>(d_nodes);
    cudaCheckError(hipGetLastError());
    cudaCheckError(hipDeviceSynchronize());

    // Copy the results back to the host
    cudaCheckError(hipMemcpy(h_nodes, d_nodes, NUM_ELEMENTS * sizeof(OctreeNode), hipMemcpyDeviceToHost));

    // Print the first 10 nodes to verify results
    for (int i = 0; i < 10; ++i) {
        std::cout << "Node " << i << ": Data = " << h_nodes[i].data << ", Children = ";
        for (int j = 0; j < 8; ++j) {
            std::cout << h_nodes[i].children[j] << " ";
        }
        std::cout << std::endl;
    }

    // Free the device memory
    cudaCheckError(hipFree(d_nodes));

    // Free the host memory
    delete[] h_nodes;

    return 0;
}