#include <iostream>
#include <hip/hip_runtime.h>


#define PARTICLES 32
#define MAX_DIST 1e9

__constant__ float c_positions[PARTICLES * 3]; // x, y, z for each particle
__constant__ float c_distances[PARTICLES * PARTICLES]; // Distance matrix

// Function to calculate squared distance between two particles
__device__ float squared_distance(int i, int j) {
    float dx = c_positions[i * 3] - c_positions[j * 3];
    float dy = c_positions[i * 3 + 1] - c_positions[j * 3 + 1];
    float dz = c_positions[i * 3 + 2] - c_positions[j * 3 + 2];
    return dx * dx + dy * dy + dz * dz;
}

// Kernel to initialize distances
__global__ void initialize_distances(float* distances) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < PARTICLES * PARTICLES) {
        int i = idx / PARTICLES;
        int j = idx % PARTICLES;
        if (i == j) {
            distances[idx] = 0.0f;
        } else {
            distances[idx] = squared_distance(i, j);
        }
    }
}

// Kernel to perform Bellman-Ford update using warp-level optimization
__global__ void bellman_ford_update(float* distances, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / PARTICLES;
    int j = idx % PARTICLES;

    if (idx < PARTICLES * PARTICLES) {
        // Use shared memory for warp-level optimization
        extern __shared__ float s_distances[];
        int lane_id = threadIdx.x % warpSize;
        int warp_id = threadIdx.x / warpSize;

        // Load data into shared memory
        s_distances[threadIdx.x] = distances[i * PARTICLES + j];
        s_distances[threadIdx.x + warpSize] = distances[i * PARTICLES + k];
        s_distances[threadIdx.x + 2 * warpSize] = distances[k * PARTICLES + j];
        __syncthreads();

        // Perform the Bellman-Ford update
        float new_dist = s_distances[threadIdx.x + warpSize] + s_distances[threadIdx.x + 2 * warpSize];
        if (new_dist < s_distances[threadIdx.x]) {
            s_distances[threadIdx.x] = new_dist;
        }
        __syncthreads();

        // Write the result back to global memory
        distances[i * PARTICLES + j] = s_distances[threadIdx.x];
    }
}

int main() {
    // Initialize host data
    float h_positions[PARTICLES * 3];
    for (int i = 0; i < PARTICLES * 3; i++) {
        h_positions[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f; // Random positions
    }

    float h_distances[PARTICLES * PARTICLES];

    // Allocate device memory
    float* d_distances;
    hipMalloc(&d_distances, PARTICLES * PARTICLES * sizeof(float));

    // Copy positions to constant memory
    hipMemcpyToSymbol(HIP_SYMBOL(c_positions), h_positions, PARTICLES * 3 * sizeof(float));

    // Initialize distances on the device
    int blockSize = 256;
    int numBlocks = (PARTICLES * PARTICLES + blockSize - 1) / blockSize;
    initialize_distances<<<numBlocks, blockSize>>>(d_distances);
    hipDeviceSynchronize();

    // Copy initial distances to host for verification
    hipMemcpy(h_distances, d_distances, PARTICLES * PARTICLES * sizeof(float), hipMemcpyDeviceToHost);

    // Perform Bellman-Ford updates
    for (int k = 0; k < PARTICLES; k++) {
        bellman_ford_update<<<numBlocks, blockSize, 3 * blockSize * sizeof(float)>>>(d_distances, k);
        hipDeviceSynchronize();
    }

    // Copy final distances to host for verification
    hipMemcpy(h_distances, d_distances, PARTICLES * PARTICLES * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_distances);

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Bellman-Ford algorithm completed successfully." << std::endl;
    return 0;
}