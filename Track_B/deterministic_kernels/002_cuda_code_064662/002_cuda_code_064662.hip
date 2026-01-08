#include <iostream>
#include <hip/hip_runtime.h>

// Define the number of elements in the octree
#define NUM_ELEMENTS 128

// Structure to represent a node in the octree
struct OctreeNode {
    int children[8]; // Indices of child nodes
    float data[4];   // Data stored at the node, padded to avoid bank conflicts
};

// Kernel to initialize the octree nodes
__global__ void initOctreeKernel(OctreeNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        // Initialize children indices to -1 (no children)
        for (int i = 0; i < 8; ++i) {
            nodes[idx].children[i] = -1;
        }
        // Initialize data with some values
        for (int i = 0; i < 4; ++i) {
            nodes[idx].data[i] = static_cast<float>(idx + i);
        }
    }
}

// Kernel to demonstrate a simple operation on the octree nodes
__global__ void processOctreeKernel(OctreeNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        // Example operation: sum the data elements
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            sum += nodes[idx].data[i];
        }
        // Store the sum back in the first data element
        nodes[idx].data[0] = sum;
    }
}

int main() {
    // Allocate memory for the octree nodes on the host
    OctreeNode* h_nodes = new OctreeNode[NUM_ELEMENTS];

    // Allocate memory for the octree nodes on the device
    OctreeNode* d_nodes;
    hipError_t err = hipMalloc((void**)&d_nodes, NUM_ELEMENTS * sizeof(OctreeNode));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize the octree nodes on the device
    int blockSize = 256;
    int numBlocks = (NUM_ELEMENTS + blockSize - 1) / blockSize;
    initOctreeKernel<<<numBlocks, blockSize>>>(d_nodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy the initialized nodes back to the host for verification
    err = hipMemcpy(h_nodes, d_nodes, NUM_ELEMENTS * sizeof(OctreeNode), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Process the octree nodes on the device
    processOctreeKernel<<<numBlocks, blockSize>>>(d_nodes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy the processed nodes back to the host for verification
    err = hipMemcpy(h_nodes, d_nodes, NUM_ELEMENTS * sizeof(OctreeNode), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_nodes);

    // Free host memory
    delete[] h_nodes;

    std::cout << "Octree processing completed successfully." << std::endl;
    return 0;
}