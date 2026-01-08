#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARTICLES 1024
#define NUM_DIMENSIONS 1024
#define MAX_ITERATIONS 100

// Define the structure for a particle
struct Particle {
    float position[NUM_DIMENSIONS];
    float velocity[NUM_DIMENSIONS];
    float pBest[NUM_DIMENSIONS];
    float fitness;
    float pBestFitness;
};

// Fitness function: Sphere function
__device__ float fitnessFunction(float* position) {
    float sum = 0.0f;
    for (int i = 0; i < NUM_DIMENSIONS; ++i) {
        sum += position[i] * position[i];
    }
    return sum;
}

// Kernel to update particle positions and velocities
__global__ void updateParticles(Particle* particles, Particle* gBest, float w, float c1, float c2, hiprandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= NUM_PARTICLES) return;

    hiprandState localState = states[idx];
    float r1 = hiprand_uniform(&localState);
    float r2 = hiprand_uniform(&localState);
    states[idx] = localState;

    for (int i = 0; i < NUM_DIMENSIONS; ++i) {
        // Update velocity
        particles[idx].velocity[i] = w * particles[idx].velocity[i] +
                                      c1 * r1 * (particles[idx].pBest[i] - particles[idx].position[i]) +
                                      c2 * r2 * (gBest->position[i] - particles[idx].position[i]);

        // Update position
        particles[idx].position[i] += particles[idx].velocity[i];

        // Apply bounds (simple example: [-5.12, 5.12])
        if (particles[idx].position[i] < -5.12f) particles[idx].position[i] = -5.12f;
        if (particles[idx].position[i] > 5.12f) particles[idx].position[i] = 5.12f;
    }

    // Evaluate fitness
    float fitness = fitnessFunction(particles[idx].position);

    // Update personal best
    if (fitness < particles[idx].pBestFitness) {
        particles[idx].pBestFitness = fitness;
        for (int i = 0; i < NUM_DIMENSIONS; ++i) {
            particles[idx].pBest[i] = particles[idx].position[i];
        }
    }
}

// Kernel to find the global best
__global__ void findGlobalBest(Particle* particles, Particle* gBest) {
    extern __shared__ Particle sharedParticles[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx >= NUM_PARTICLES) return;

    // Load particle data into shared memory
    sharedParticles[tid] = particles[idx];
    __syncthreads();

    // Find the minimum fitness in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sharedParticles[tid + s].pBestFitness < sharedParticles[tid].pBestFitness) {
                sharedParticles[tid] = sharedParticles[tid + s];
            }
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        atomicMin(reinterpret_cast<unsigned int*>(&gBest->pBestFitness), __float_as_uint(sharedParticles[0].pBestFitness));
        if (__float_as_uint(gBest->pBestFitness) == __float_as_uint(sharedParticles[0].pBestFitness)) {
            for (int i = 0; i < NUM_DIMENSIONS; ++i) {
                gBest->pBest[i] = sharedParticles[0].pBest[i];
            }
        }
    }
}

// Initialize particles on the device
__global__ void initializeParticles(Particle* particles, hiprandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= NUM_PARTICLES) return;

    hiprandState localState = states[idx];
    for (int i = 0; i < NUM_DIMENSIONS; ++i) {
        particles[idx].position[i] = hiprand_uniform(&localState) * 10.24f - 5.12f;
        particles[idx].velocity[i] = hiprand_uniform(&localState) * 0.1f - 0.05f;
        particles[idx].pBest[i] = particles[idx].position[i];
    }
    particles[idx].fitness = fitnessFunction(particles[idx].position);
    particles[idx].pBestFitness = particles[idx].fitness;
    states[idx] = localState;
}

// Initialize curand states
__global__ void initCurandStates(hiprandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        hiprand_init(seed, idx, 0, &states[idx]);
    }
}

int main() {
    const int blockSize = 256;
    const int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;

    Particle* d_particles;
    Particle* d_gBest;
    hiprandState* d_states;

    // Allocate memory on the device
    hipError_t err = hipMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle));
    if (err != hipSuccess) {
        std::cerr << "Error allocating particles: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&d_gBest, sizeof(Particle));
    if (err != hipSuccess) {
        std::cerr << "Error allocating gBest: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&d_states, NUM_PARTICLES * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Error allocating states: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize curand states
    initCurandStates<<<numBlocks, blockSize>>>(d_states, 12345UL);
    hipDeviceSynchronize();

    // Initialize particles
    initializeParticles<<<numBlocks, blockSize>>>(d_particles, d_states);
    hipDeviceSynchronize();

    // Initialize global best
    Particle h_gBest;
    for (int i = 0; i < NUM_DIMENSIONS; ++i) {
        h_gBest.pBest[i] = 10000.0f; // Arbitrarily large number
    }
    h_gBest.pBestFitness = 10000.0f;
    hipMemcpy(d_gBest, &h_gBest, sizeof(Particle), hipMemcpyHostToDevice);

    // PSO parameters
    float w = 0.729; // Inertia weight
    float c1 = 1.49445; // Cognitive parameter
    float c2 = 1.49445; // Social parameter

    // PSO iterations
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        updateParticles<<<numBlocks, blockSize>>>(d_particles, d_gBest, w, c1, c2, d_states);
        hipDeviceSynchronize();

        findGlobalBest<<<numBlocks, blockSize, blockSize * sizeof(Particle)>>>(d_particles, d_gBest);
        hipDeviceSynchronize();
    }

    // Copy global best back to host
    hipMemcpy(&h_gBest, d_gBest, sizeof(Particle), hipMemcpyDeviceToHost);

    // Output the result
    std::cout << "Global Best Fitness: " << h_gBest.pBestFitness << std::endl;

    // Free device memory
    hipFree(d_particles);
    hipFree(d_gBest);
    hipFree(d_states);

    return 0;
}