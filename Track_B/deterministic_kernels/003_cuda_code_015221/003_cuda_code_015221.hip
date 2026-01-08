// 2048D Fluid Simulation using Particle-in-Cell Method
// Multi-GPU Scaling with Concurrent Kernel Execution
// Target Architecture: Volta

#include <iostream>
#include <hip/hip_runtime.h>
#include <vector>
#include <hiprand/hiprand_kernel.h>

#define BLOCK_SIZE 256
#define NUM_PARTICLES 2048
#define NUM_GPUS 2

// Particle structure
struct Particle {
    float4 pos;
    float4 vel;
};

// Initialize random number generator states
__global__ void initRandomStates(hiprandState *states, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    hiprand_init(seed, id, 0, &states[id]);
}

// Initialize particles with random positions and velocities
__global__ void initParticles(Particle *particles, hiprandState *states) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < NUM_PARTICLES) {
        hiprandState localState = states[id];
        particles[id].pos = make_float4(hiprand_uniform(&localState) * 10.0f,
                                        hiprand_uniform(&localState) * 10.0f,
                                        hiprand_uniform(&localState) * 10.0f,
                                        hiprand_uniform(&localState) * 10.0f);
        particles[id].vel = make_float4(hiprand_uniform(&localState) * 2.0f - 1.0f,
                                        hiprand_uniform(&localState) * 2.0f - 1.0f,
                                        hiprand_uniform(&localState) * 2.0f - 1.0f,
                                        hiprand_uniform(&localState) * 2.0f - 1.0f);
        states[id] = localState;
    }
}

// Kernel to update particle positions and velocities
__global__ void updateParticles(Particle *particles, float dt) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < NUM_PARTICLES) {
        // Simple Euler integration for demonstration
        particles[id].pos.x += particles[id].vel.x * dt;
        particles[id].pos.y += particles[id].vel.y * dt;
        particles[id].pos.z += particles[id].vel.z * dt;
        particles[id].pos.w += particles[id].vel.w * dt;
    }
}

// Check for CUDA errors
void checkCudaError(hipError_t err, const char *file, int line) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)

int main() {
    int numDevices;
    hipGetDeviceCount(&numDevices);
    if (numDevices < NUM_GPUS) {
        std::cerr << "Error: Not enough GPUs available." << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate memory and initialize random states on each GPU
    std::vector<Particle*> d_particles(NUM_GPUS);
    std::vector<hiprandState*> d_randomStates(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);
        CHECK_CUDA_ERROR(hipMalloc(&d_particles[i], NUM_PARTICLES * sizeof(Particle)));
        CHECK_CUDA_ERROR(hipMalloc(&d_randomStates[i], NUM_PARTICLES * sizeof(hiprandState)));
        initRandomStates<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_randomStates[i], time(NULL));
        CHECK_CUDA_ERROR(hipGetLastError());
        CHECK_CUDA_ERROR(hipDeviceSynchronize());
    }

    // Initialize particles on each GPU
    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);
        initParticles<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles[i], d_randomStates[i]);
        CHECK_CUDA_ERROR(hipGetLastError());
        CHECK_CUDA_ERROR(hipDeviceSynchronize());
    }

    // Simulate for 100 time steps
    float dt = 0.01f;
    for (int step = 0; step < 100; ++step) {
        for (int i = 0; i < NUM_GPUS; ++i) {
            hipSetDevice(i);
            updateParticles<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles[i], dt);
            CHECK_CUDA_ERROR(hipGetLastError());
        }
        CHECK_CUDA_ERROR(hipDeviceSynchronize());
    }

    // Free allocated memory on each GPU
    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);
        CHECK_CUDA_ERROR(hipFree(d_particles[i]));
        CHECK_CUDA_ERROR(hipFree(d_randomStates[i]));
    }

    std::cout << "Simulation completed successfully." << std::endl;
    return EXIT_SUCCESS;
}