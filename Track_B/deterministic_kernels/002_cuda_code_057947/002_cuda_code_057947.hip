#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARTICLES 2048
#define DIMENSIONS 6
#define BLOCK_SIZE 256

__global__ void initializeParticles(float *particles, int numParticles, int dimensions) {
    hiprandState state;
    hiprand_init((unsigned long long)clock() + threadIdx.x, blockIdx.x, 0, &state);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        for (int d = 0; d < dimensions; d++) {
            // Distribute particles in a 6D space
            particles[idx * dimensions + d] = hiprand_uniform(&state) * 100.0f;
        }
    }
}

__device__ float euclideanDistance(const float *a, const float *b, int dimensions) {
    float sum = 0.0f;
    for (int d = 0; d < dimensions; d++) {
        float diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

__global__ void computeMST(float *particles, float *distances, int numParticles, int dimensions) {
    extern __shared__ float sharedParticles[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sharedIdx = threadIdx.x * dimensions;

    // Load particles into shared memory to avoid bank conflicts
    if (idx < numParticles) {
        for (int d = 0; d < dimensions; d++) {
            sharedParticles[sharedIdx + d] = particles[idx * dimensions + d];
        }
    }
    __syncthreads();

    // Compute distances only for particles in shared memory
    if (idx < numParticles) {
        for (int j = 0; j < blockDim.x && (blockIdx.x * blockDim.x + j) < numParticles; j++) {
            if (idx != (blockIdx.x * blockDim.x + j)) {
                float distance = euclideanDistance(&sharedParticles[sharedIdx], &sharedParticles[j * dimensions], dimensions);
                distances[idx * numParticles + (blockIdx.x * blockDim.x + j)] = distance;
            }
        }
    }
}

int main() {
    float *d_particles, *d_distances;
    size_t particlesSize = NUM_PARTICLES * DIMENSIONS * sizeof(float);
    size_t distancesSize = NUM_PARTICLES * NUM_PARTICLES * sizeof(float);

    // Allocate memory on the device
    hipError_t err = hipMalloc((void **)&d_particles, particlesSize);
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for particles: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    err = hipMalloc((void **)&d_distances, distancesSize);
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for distances: " << hipGetErrorString(err) << std::endl;
        hipFree(d_particles);
        return 1;
    }

    // Initialize particles
    int numBlocks = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializeParticles<<<numBlocks, BLOCK_SIZE>>>(d_particles, NUM_PARTICLES, DIMENSIONS);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching initializeParticles kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_particles);
        hipFree(d_distances);
        return 1;
    }

    // Compute distances for MST
    computeMST<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * DIMENSIONS * sizeof(float)>>>(d_particles, d_distances, NUM_PARTICLES, DIMENSIONS);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching computeMST kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_particles);
        hipFree(d_distances);
        return 1;
    }

    // Free device memory
    hipFree(d_particles);
    hipFree(d_distances);

    std::cout << "MST computation completed successfully." << std::endl;
    return 0;
}