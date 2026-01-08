#include <iostream>
#include <hip/hip_runtime.h>


// Define the number of nodes
#define NUM_NODES 16384

// Kernel to initialize communities
__global__ void initializeCommunities(int* communities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        communities[idx] = idx; // Each node starts in its own community
    }
}

// Kernel to compute the change in modularity for a node
__global__ void computeDeltaQ(int* communities, int* edges, int* weights, float* deltaQ, int* degrees) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        int oldCommunity = communities[idx];
        float oldDeltaQ = 0.0f;
        float newDeltaQ = 0.0f;
        int totalWeight = 0;

        // Sum weights of edges connected to node idx
        for (int j = 0; j < degrees[idx]; ++j) {
            int neighbor = edges[idx * NUM_NODES + j];
            int weight = weights[idx * NUM_NODES + j];
            totalWeight += weight;
            if (communities[neighbor] == oldCommunity) {
                oldDeltaQ += 2.0f * weight - static_cast<float>(degrees[idx] * degrees[neighbor]) / (2 * totalWeight);
            }
        }

        // Try moving node to each community and compute newDeltaQ
        for (int c = 0; c < NUM_NODES; ++c) {
            if (c == oldCommunity) continue;
            newDeltaQ = 0.0f;
            for (int j = 0; j < degrees[idx]; ++j) {
                int neighbor = edges[idx * NUM_NODES + j];
                int weight = weights[idx * NUM_NODES + j];
                if (communities[neighbor] == c) {
                    newDeltaQ += 2.0f * weight - static_cast<float>(degrees[idx] * degrees[neighbor]) / (2 * totalWeight);
                }
            }
            if (newDeltaQ > oldDeltaQ) {
                communities[idx] = c;
                deltaQ[idx] = newDeltaQ;
                break;
            }
        }
    }
}

// Main function to demonstrate the usage of the kernel
int main() {
    // Allocate memory for graph data on the host
    int* h_edges = new int[NUM_NODES * NUM_NODES];
    int* h_weights = new int[NUM_NODES * NUM_NODES];
    int* h_degrees = new int[NUM_NODES];
    int* h_communities = new int[NUM_NODES];
    float* h_deltaQ = new float[NUM_NODES];

    // Initialize graph data (dummy data for demonstration)
    for (int i = 0; i < NUM_NODES; ++i) {
        h_degrees[i] = 0;
        for (int j = 0; j < NUM_NODES; ++j) {
            h_edges[i * NUM_NODES + j] = 0;
            h_weights[i * NUM_NODES + j] = 0;
        }
    }
    // Example: connect each node to its next neighbor
    for (int i = 0; i < NUM_NODES - 1; ++i) {
        h_edges[i * NUM_NODES + (i + 1)] = i + 1;
        h_edges[(i + 1) * NUM_NODES + i] = i;
        h_weights[i * NUM_NODES + (i + 1)] = 1;
        h_weights[(i + 1) * NUM_NODES + i] = 1;
        h_degrees[i]++;
        h_degrees[i + 1]++;
    }

    // Allocate memory for graph data on the device
    int* d_edges, *d_weights, *d_degrees, *d_communities;
    float* d_deltaQ;
    hipMalloc(&d_edges, NUM_NODES * NUM_NODES * sizeof(int));
    hipMalloc(&d_weights, NUM_NODES * NUM_NODES * sizeof(int));
    hipMalloc(&d_degrees, NUM_NODES * sizeof(int));
    hipMalloc(&d_communities, NUM_NODES * sizeof(int));
    hipMalloc(&d_deltaQ, NUM_NODES * sizeof(float));

    // Copy graph data from host to device
    hipMemcpy(d_edges, h_edges, NUM_NODES * NUM_NODES * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_weights, h_weights, NUM_NODES * NUM_NODES * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_degrees, h_degrees, NUM_NODES * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_communities, h_communities, NUM_NODES * sizeof(int), hipMemcpyHostToDevice);

    // Initialize communities
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_NODES + threadsPerBlock - 1) / threadsPerBlock;
    initializeCommunities<<<blocksPerGrid, threadsPerBlock>>>(d_communities);
    hipDeviceSynchronize();

    // Check for errors in kernel execution
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Perform community detection iterations
    bool improvement = true;
    int maxIterations = 10; // Limit iterations for demonstration
    for (int iter = 0; iter < maxIterations && improvement; ++iter) {
        improvement = false;
        // Compute deltaQ for each node
        computeDeltaQ<<<blocksPerGrid, threadsPerBlock>>>(d_communities, d_edges, d_weights, d_deltaQ, d_degrees);
        hipDeviceSynchronize();

        // Check for errors in kernel execution
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Copy deltaQ from device to host to check for improvements
        hipMemcpy(h_deltaQ, d_deltaQ, NUM_NODES * sizeof(float), hipMemcpyDeviceToHost);
        for (int i = 0; i < NUM_NODES; ++i) {
            if (h_deltaQ[i] > 0.0f) {
                improvement = true;
                break;
            }
        }
    }

    // Copy communities from device to host
    hipMemcpy(h_communities, d_communities, NUM_NODES * sizeof(int), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_edges);
    hipFree(d_weights);
    hipFree(d_degrees);
    hipFree(d_communities);
    hipFree(d_deltaQ);

    // Free host memory
    delete[] h_edges;
    delete[] h_weights;
    delete[] h_degrees;
    delete[] h_communities;
    delete[] h_deltaQ;

    std::cout << "Community detection completed." << std::endl;
    return 0;
}