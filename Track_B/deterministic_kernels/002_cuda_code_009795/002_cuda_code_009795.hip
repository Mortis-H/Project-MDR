#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARTICLES 8192
#define INF 1e9f

// Kernel to initialize random positions and distances
__global__ void initParticles(float *positions, float *distances, hiprandState *states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        hiprandState localState = states[idx];
        hiprand_init(seed, idx, 0, &localState);
        positions[idx * 2] = hiprand_uniform(&localState) * 100.0f;
        positions[idx * 2 + 1] = hiprand_uniform(&localState) * 100.0f;
        distances[idx] = (idx == 0) ? 0.0f : INF;
        states[idx] = localState;
    }
}

// Kernel to compute distances using Bellman-Ford algorithm
__global__ void bellmanFord(float *positions, float *distances, int iter) {
    extern __shared__ float sharedDistances[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load distances into shared memory
    if (idx < NUM_PARTICLES) {
        sharedDistances[tid] = distances[idx];
    }
    __syncthreads();

    // Calculate the distance between particles
    if (idx < NUM_PARTICLES) {
        float x1 = positions[idx * 2];
        float y1 = positions[idx * 2 + 1];
        for (int j = 0; j < blockDim.x; ++j) {
            if (tid != j) {
                float x2 = positions[j * 2];
                float y2 = positions[j * 2 + 1];
                float dist = sqrtf((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
                if (sharedDistances[j] + dist < sharedDistances[tid]) {
                    sharedDistances[tid] = sharedDistances[j] + dist;
                }
            }
        }
    }
    __syncthreads();

    // Write updated distances back to global memory
    if (idx < NUM_PARTICLES) {
        distances[idx] = sharedDistances[tid];
    }
}

int main() {
    // Allocate and initialize device memory
    float *d_positions, *d_distances;
    hiprandState *d_states;
    hipMalloc((void**)&d_positions, NUM_PARTICLES * 2 * sizeof(float));
    hipMalloc((void**)&d_distances, NUM_PARTICLES * sizeof(float));
    hipMalloc((void**)&d_states, NUM_PARTICLES * sizeof(hiprandState));

    // Initialize particles on the device
    int blockSize = 256;
    int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;
    initParticles<<<numBlocks, blockSize>>>(d_positions, d_distances, d_states, 1234);
    hipDeviceSynchronize();

    // Perform Bellman-Ford algorithm
    for (int iter = 0; iter < NUM_PARTICLES - 1; ++iter) {
        bellmanFord<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_positions, d_distances, iter);
        hipDeviceSynchronize();
    }

    // Copy results back to host
    thrust::device_ptr<float> d_distances_ptr(d_distances);
    thrust::host_vector<float> h_distances(d_distances_ptr, d_distances_ptr + NUM_PARTICLES);

    // Print the distances from the source particle
    for (int i = 0; i < 10; ++i) {
        std::cout << "Distance to particle " << i << ": " << h_distances[i] << std::endl;
    }

    // Free device memory
    hipFree(d_positions);
    hipFree(d_distances);
    hipFree(d_states);

    return 0;
}