#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 1024
#define BLOCK_SIZE 256

// Structure to hold particle data
struct Particle {
    float4 pos;  // x, y, z, w (w can be used for mass or other properties)
    float4 vel;  // vx, vy, vz, w (w can be used for force or other properties)
};

// CUDA kernel to perform Verlet integration step
__global__ void verletIntegration(Particle *d_particles, float dt, float4 *d_forces, float4 box) {
    extern __shared__ float4 shared_pos[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load positions into shared memory to avoid bank conflicts
    shared_pos[threadIdx.x] = d_particles[idx].pos;
    __syncthreads();

    // Calculate new positions using Verlet integration
    float4 newPos;
    newPos.x = shared_pos[threadIdx.x].x + d_particles[idx].vel.x * dt + d_forces[idx].x * dt * dt * 0.5f;
    newPos.y = shared_pos[threadIdx.x].y + d_particles[idx].vel.y * dt + d_forces[idx].y * dt * dt * 0.5f;
    newPos.z = shared_pos[threadIdx.x].z + d_particles[idx].vel.z * dt + d_forces[idx].z * dt * dt * 0.5f;
    newPos.w = shared_pos[threadIdx.x].w; // Copy mass or other properties

    // Apply periodic boundary conditions
    newPos.x = fmodf(newPos.x - box.x, box.y - box.x) + box.x;
    newPos.y = fmodf(newPos.y - box.x, box.y - box.x) + box.x;
    newPos.z = fmodf(newPos.z - box.x, box.y - box.x) + box.x;

    // Store new positions
    d_particles[idx].pos = newPos;

    // Update velocities
    float4 newVel;
    newVel.x = d_particles[idx].vel.x + d_forces[idx].x * dt * 0.5f;
    newVel.y = d_particles[idx].vel.y + d_forces[idx].y * dt * 0.5f;
    newVel.z = d_particles[idx].vel.z + d_forces[idx].z * dt * 0.5f;
    newVel.w = d_particles[idx].vel.w; // Copy mass or other properties

    d_particles[idx].vel = newVel;
}

// Initialize random positions and velocities for particles
__global__ void initParticles(Particle *d_particles, hiprandState *states, float4 box) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Initialize random number generator
    hiprandState localState = states[idx];
    hiprand_init(idx, 0, 0, &localState);

    // Initialize positions and velocities
    d_particles[idx].pos.x = hiprand_uniform(&localState) * (box.y - box.x) + box.x;
    d_particles[idx].pos.y = hiprand_uniform(&localState) * (box.y - box.x) + box.x;
    d_particles[idx].pos.z = hiprand_uniform(&localState) * (box.y - box.x) + box.x;
    d_particles[idx].pos.w = 1.0f; // Mass

    d_particles[idx].vel.x = (hiprand_uniform(&localState) - 0.5f) * 0.1f;
    d_particles[idx].vel.y = (hiprand_uniform(&localState) - 0.5f) * 0.1f;
    d_particles[idx].vel.z = (hiprand_uniform(&localState) - 0.5f) * 0.1f;
    d_particles[idx].vel.w = 0.0f; // Force

    // Update global state
    states[idx] = localState;
}

// Initialize random forces for particles
__global__ void initForces(float4 *d_forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    d_forces[idx].x = 0.0f;
    d_forces[idx].y = 0.0f;
    d_forces[idx].z = 0.0f;
    d_forces[idx].w = 0.0f;
}

int main() {
    Particle *d_particles;
    float4 *d_forces;
    hiprandState *d_states;
    float4 box = make_float4(0.0f, 10.0f, 0.0f, 0.0f); // Define a cubic box

    // Allocate memory on the device
    hipMalloc(&d_particles, N * sizeof(Particle));
    hipMalloc(&d_forces, N * sizeof(float4));
    hipMalloc(&d_states, N * sizeof(hiprandState));

    // Initialize particles and forces
    initParticles<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles, d_states, box);
    hipDeviceSynchronize();

    initForces<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_forces);
    hipDeviceSynchronize();

    // Time step for integration
    float dt = 0.01f;

    // Perform Verlet integration
    verletIntegration<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float4)>>>(d_particles, dt, d_forces, box);
    hipDeviceSynchronize();

    // Free device memory
    hipFree(d_particles);
    hipFree(d_forces);
    hipFree(d_states);

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Molecular dynamics simulation completed successfully." << std::endl;
    return 0;
}