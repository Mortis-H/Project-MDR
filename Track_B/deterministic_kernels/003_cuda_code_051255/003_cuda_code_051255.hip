#include <iostream>
#include <hip/hip_runtime.h>


// Define constants
#define NUM_NODES 2048
#define MAX_THREADS_PER_BLOCK 1024
#define INF 1e9

// Kernel to initialize communities
__global__ void initializeCommunities(int *communities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        communities[idx] = idx; // Each node starts in its own community
    }
}

// Kernel to calculate modularity gain for a node moving to a new community
__global__ void calculateModularityGain(int *communities, float *weights, float *degrees, float *gain, int node) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        if (idx != node && communities[idx] != communities[node]) {
            gain[idx] = (weights[node * NUM_NODES + idx] - degrees[node] * degrees[idx] / (2 * INF));
        } else {
            gain[idx] = -INF;
        }
    }
}

// Kernel to update community for a node based on maximum modularity gain
__global__ void updateCommunity(int *communities, float *gain, int node) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        if (gain[idx] > gain[node]) {
            node = idx;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        communities[node] = communities[node];
    }
}

// Main function to demonstrate the usage of the kernels
int main() {
    // Allocate memory on host
    int *h_communities = new int[NUM_NODES];
    float *h_weights = new float[NUM_NODES * NUM_NODES];
    float *h_degrees = new float[NUM_NODES];

    // Initialize host data
    for (int i = 0; i < NUM_NODES; i++) {
        h_communities[i] = i;
        h_degrees[i] = 0.0f;
        for (int j = 0; j < NUM_NODES; j++) {
            h_weights[i * NUM_NODES + j] = (i == j) ? 0.0f : (float)(rand() % 10) / 10.0f; // Random weights
            h_degrees[i] += h_weights[i * NUM_NODES + j];
        }
    }

    // Allocate memory on device
    int *d_communities;
    float *d_weights, *d_degrees, *d_gain;
    hipMalloc((void **)&d_communities, NUM_NODES * sizeof(int));
    hipMalloc((void **)&d_weights, NUM_NODES * NUM_NODES * sizeof(float));
    hipMalloc((void **)&d_degrees, NUM_NODES * sizeof(float));
    hipMalloc((void **)&d_gain, NUM_NODES * sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_communities, h_communities, NUM_NODES * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_weights, h_weights, NUM_NODES * NUM_NODES * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_degrees, h_degrees, NUM_NODES * sizeof(float), hipMemcpyHostToDevice);

    // Launch kernel to initialize communities
    int blockSize = MAX_THREADS_PER_BLOCK;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;
    initializeCommunities<<<numBlocks, blockSize>>>(d_communities);
    hipDeviceSynchronize();

    // Simulate community detection iterations
    for (int iter = 0; iter < 10; iter++) {
        for (int node = 0; node < NUM_NODES; node++) {
            // Calculate modularity gain for each node
            calculateModularityGain<<<numBlocks, blockSize>>>(d_communities, d_weights, d_degrees, d_gain, node);
            hipDeviceSynchronize();

            // Update community for the node based on maximum modularity gain
            updateCommunity<<<1, NUM_NODES>>>(d_communities, d_gain, node);
            hipDeviceSynchronize();
        }
    }

    // Copy results back to host
    hipMemcpy(h_communities, d_communities, NUM_NODES * sizeof(int), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_communities);
    hipFree(d_weights);
    hipFree(d_degrees);
    hipFree(d_gain);

    // Free host memory
    delete[] h_communities;
    delete[] h_weights;
    delete[] h_degrees;

    // Check for any errors launching the kernel
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Community detection completed successfully." << std::endl;
    return 0;
}