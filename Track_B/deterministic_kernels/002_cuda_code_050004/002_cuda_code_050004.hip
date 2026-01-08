#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

#define NUM_PARTICLES 2048
#define NUM_ITERATIONS 100
#define DAMPING_FACTOR 0.85

__global__ void initializeRanks(float *ranks, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        ranks[idx] = 1.0f / numParticles;
    }
}

__global__ void updateRanks(const float *currentRanks, float *newRanks, const int *adjacencyList, const int *rowOffsets, int numParticles, float dampingFactor) {
    extern __shared__ float sharedRanks[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        sharedRanks[threadIdx.x] = currentRanks[idx];
        __syncthreads();

        float rankSum = 0.0f;
        int start = rowOffsets[idx];
        int end = rowOffsets[idx + 1];
        for (int j = start; j < end; ++j) {
            int neighbor = adjacencyList[j];
            rankSum += sharedRanks[neighbor % blockDim.x];
        }
        float newRank = (1.0f - dampingFactor) / numParticles + dampingFactor * rankSum;
        atomicAdd(&newRanks[idx], newRank);
    }
}

int main() {
    // Initialize random number generator
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(0, NUM_PARTICLES - 1);

    // Allocate memory for ranks
    float *d_currentRanks, *d_newRanks;
    hipMalloc(&d_currentRanks, NUM_PARTICLES * sizeof(float));
    hipMalloc(&d_newRanks, NUM_PARTICLES * sizeof(float));

    // Initialize ranks
    int blockSize = 256;
    int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;
    initializeRanks<<<numBlocks, blockSize>>>(d_currentRanks, NUM_PARTICLES);
    hipDeviceSynchronize();

    // Initialize adjacency list and row offsets
    int *d_adjacencyList, *d_rowOffsets;
    int *h_rowOffsets = new int[NUM_PARTICLES + 1];
    h_rowOffsets[0] = 0;
    int totalEdges = 0;

    for (int i = 0; i < NUM_PARTICLES; ++i) {
        int numEdges = dist(rng) % 10; // Random number of edges between 0 and 9
        h_rowOffsets[i + 1] = h_rowOffsets[i] + numEdges;
        totalEdges += numEdges;
    }

    int *h_adjacencyList = new int[totalEdges];
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        for (int j = h_rowOffsets[i]; j < h_rowOffsets[i + 1]; ++j) {
            h_adjacencyList[j] = dist(rng) % NUM_PARTICLES;
        }
    }

    hipMalloc(&d_adjacencyList, totalEdges * sizeof(int));
    hipMalloc(&d_rowOffsets, (NUM_PARTICLES + 1) * sizeof(int));
    hipMemcpy(d_adjacencyList, h_adjacencyList, totalEdges * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_rowOffsets, h_rowOffsets, (NUM_PARTICLES + 1) * sizeof(int), hipMemcpyHostToDevice);

    // PageRank iterations
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        hipMemset(d_newRanks, 0, NUM_PARTICLES * sizeof(float));
        updateRanks<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_currentRanks, d_newRanks, d_adjacencyList, d_rowOffsets, NUM_PARTICLES, DAMPING_FACTOR);
        hipDeviceSynchronize();

        // Swap current and new ranks
        float *temp = d_currentRanks;
        d_currentRanks = d_newRanks;
        d_newRanks = temp;
    }

    // Copy final ranks to host and print
    float *h_ranks = new float[NUM_PARTICLES];
    hipMemcpy(h_ranks, d_currentRanks, NUM_PARTICLES * sizeof(float), hipMemcpyDeviceToHost);

    std::cout << "Final PageRank values:\n";
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        std::cout << "Particle " << i << ": " << h_ranks[i] << "\n";
    }

    // Free memory
    hipFree(d_currentRanks);
    hipFree(d_newRanks);
    hipFree(d_adjacencyList);
    hipFree(d_rowOffsets);
    delete[] h_adjacencyList;
    delete[] h_rowOffsets;
    delete[] h_ranks;

    return 0;
}