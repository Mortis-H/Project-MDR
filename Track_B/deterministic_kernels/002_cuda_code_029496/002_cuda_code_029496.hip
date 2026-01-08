#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARTICLES 128
#define DIMENSIONS 6
#define MAX_ITERATIONS 1000

// Define a simple heuristic function (Manhattan distance)
__device__ float heuristic(const float* a, const float* b) {
    float sum = 0.0f;
    for (int i = 0; i < DIMENSIONS; ++i) {
        sum += fabs(a[i] - b[i]);
    }
    return sum;
}

// Structure to hold particle data
struct Particle {
    float position[DIMENSIONS];
    float g_score; // Cost from start to current node
    float f_score; // Estimated total cost from start to goal through current node
    int parent;    // Index of the parent node
};

// Kernel to initialize particles
__global__ void initializeParticles(Particle* particles, const float* start, const float* goal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        for (int i = 0; i < DIMENSIONS; ++i) {
            particles[idx].position[i] = start[i] + (goal[i] - start[i]) * (idx % 2);
        }
        particles[idx].g_score = 0.0f;
        particles[idx].f_score = heuristic(particles[idx].position, goal);
        particles[idx].parent = -1;
    }
}

// Kernel to update particles based on A* logic
__global__ void updateParticles(Particle* particles, const float* goal) {
    extern __shared__ Particle shared_particles[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < NUM_PARTICLES) {
        shared_particles[tid] = particles[idx];
    }
    __syncthreads();

    if (tid == 0) {
        // Find the particle with the lowest f_score
        int min_idx = 0;
        for (int i = 1; i < blockDim.x; ++i) {
            if (shared_particles[i].f_score < shared_particles[min_idx].f_score) {
                min_idx = i;
            }
        }
        // Update the best particle's position (simple movement towards goal)
        for (int i = 0; i < DIMENSIONS; ++i) {
            shared_particles[min_idx].position[i] += (goal[i] - shared_particles[min_idx].position[i]) * 0.01f;
        }
        // Recalculate scores
        shared_particles[min_idx].g_score += 0.01f;
        shared_particles[min_idx].f_score = shared_particles[min_idx].g_score + heuristic(shared_particles[min_idx].position, goal);
    }
    __syncthreads();

    if (idx < NUM_PARTICLES) {
        particles[idx] = shared_particles[tid];
    }
}

// Function to check CUDA errors
void checkCudaError(hipError_t error, const char* message) {
    if (error != hipSuccess) {
        std::cerr << "CUDA Error: " << message << " (" << hipGetErrorString(error) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Initialize random number generator
    hiprandState* devStates;
    hipMalloc(&devStates, NUM_PARTICLES * sizeof(hiprandState));
    hiprandState* d_states = devStates;
    hipDeviceSynchronize();

    // Allocate memory for particles
    Particle* d_particles;
    hipMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle));

    // Allocate and initialize start and goal positions
    float* d_start, *d_goal;
    hipMalloc(&d_start, DIMENSIONS * sizeof(float));
    hipMalloc(&d_goal, DIMENSIONS * sizeof(float));
    float h_start[DIMENSIONS] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float h_goal[DIMENSIONS] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    hipMemcpy(d_start, h_start, DIMENSIONS * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_goal, h_goal, DIMENSIONS * sizeof(float), hipMemcpyHostToDevice);

    // Initialize particles
    initializeParticles<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_particles, d_start, d_goal);
    hipDeviceSynchronize();
    checkCudaError(hipGetLastError(), "initializeParticles kernel launch failed");

    // Update particles in a loop
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        updateParticles<<<(NUM_PARTICLES + 255) / 256, 256, 256 * sizeof(Particle)>>>(d_particles, d_goal);
        hipDeviceSynchronize();
        checkCudaError(hipGetLastError(), "updateParticles kernel launch failed");
    }

    // Free memory
    hipFree(d_particles);
    hipFree(d_start);
    hipFree(d_goal);
    hipFree(devStates);

    std::cout << "A* search completed successfully." << std::endl;
    return 0;
}