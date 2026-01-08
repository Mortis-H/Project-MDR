#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define GRID_SIZE 1024
#define BLOCK_SIZE 16
#define NUM_PARTICLES (GRID_SIZE * GRID_SIZE)

__constant__ float d_gravity = 9.81f;
__constant__ float d_damping = 0.01f;
__constant__ float d_stiffness = 0.1f;

struct Particle {
    float2 pos;
    float2 vel;
};

__global__ void initializeParticles(Particle *particles, hiprandState *states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    hiprandState localState = states[idx];
    hiprand_init(seed, idx, 0, &localState);

    // Random initial positions and velocities
    particles[idx].pos.x = hiprand_uniform(&localState) * GRID_SIZE;
    particles[idx].pos.y = hiprand_uniform(&localState) * GRID_SIZE;
    particles[idx].vel.x = hiprand_uniform(&localState) - 0.5f;
    particles[idx].vel.y = hiprand_uniform(&localState) - 0.5f;

    states[idx] = localState;
}

__device__ float2 computeForce(Particle *particles, int idx) {
    float2 force = {0.0f, d_gravity};

    // Simple spring force with neighbors
    int x = idx % GRID_SIZE;
    int y = idx / GRID_SIZE;

    // Check left neighbor
    if (x > 0) {
        int leftIdx = idx - 1;
        float2 diff = {particles[idx].pos.x - particles[leftIdx].pos.x, particles[idx].pos.y - particles[leftIdx].pos.y};
        float dist = sqrtf(diff.x * diff.x + diff.y * diff.y);
        float2 dir = {diff.x / dist, diff.y / dist};
        force.x += d_stiffness * (1.0f - dist) * dir.x;
        force.y += d_stiffness * (1.0f - dist) * dir.y;
    }

    // Check right neighbor
    if (x < GRID_SIZE - 1) {
        int rightIdx = idx + 1;
        float2 diff = {particles[idx].pos.x - particles[rightIdx].pos.x, particles[idx].pos.y - particles[rightIdx].pos.y};
        float dist = sqrtf(diff.x * diff.x + diff.y * diff.y);
        float2 dir = {diff.x / dist, diff.y / dist};
        force.x += d_stiffness * (1.0f - dist) * dir.x;
        force.y += d_stiffness * (1.0f - dist) * dir.y;
    }

    // Check top neighbor
    if (y > 0) {
        int topIdx = idx - GRID_SIZE;
        float2 diff = {particles[idx].pos.x - particles[topIdx].pos.x, particles[idx].pos.y - particles[topIdx].pos.y};
        float dist = sqrtf(diff.x * diff.x + diff.y * diff.y);
        float2 dir = {diff.x / dist, diff.y / dist};
        force.x += d_stiffness * (1.0f - dist) * dir.x;
        force.y += d_stiffness * (1.0f - dist) * dir.y;
    }

    // Check bottom neighbor
    if (y < GRID_SIZE - 1) {
        int bottomIdx = idx + GRID_SIZE;
        float2 diff = {particles[idx].pos.x - particles[bottomIdx].pos.x, particles[idx].pos.y - particles[bottomIdx].pos.y};
        float dist = sqrtf(diff.x * diff.x + diff.y * diff.y);
        float2 dir = {diff.x / dist, diff.y / dist};
        force.x += d_stiffness * (1.0f - dist) * dir.x;
        force.y += d_stiffness * (1.0f - dist) * dir.y;
    }

    return force;
}

__global__ void updateParticles(Particle *particles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    float2 force = computeForce(particles, idx);

    // Update velocity with damping
    particles[idx].vel.x += force.x * dt;
    particles[idx].vel.y += force.y * dt;
    particles[idx].vel.x *= (1.0f - d_damping);
    particles[idx].vel.y *= (1.0f - d_damping);

    // Update position
    particles[idx].pos.x += particles[idx].vel.x * dt;
    particles[idx].pos.y += particles[idx].vel.y * dt;

    // Boundary conditions (simple wrap-around)
    particles[idx].pos.x = fmodf(particles[idx].pos.x + GRID_SIZE, GRID_SIZE);
    particles[idx].pos.y = fmodf(particles[idx].pos.y + GRID_SIZE, GRID_SIZE);
}

int main() {
    Particle *d_particles;
    hiprandState *d_states;

    hipError_t err = hipMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for particles: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc(&d_states, NUM_PARTICLES * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for states: " << hipGetErrorString(err) << std::endl;
        hipFree(d_particles);
        return -1;
    }

    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((NUM_PARTICLES + threadsPerBlock.x - 1) / threadsPerBlock.x);

    initializeParticles<<<blocksPerGrid, threadsPerBlock>>>(d_particles, d_states, time(NULL));
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize particles: " << hipGetErrorString(err) << std::endl;
        hipFree(d_particles);
        hipFree(d_states);
        return -1;
    }

    float dt = 0.01f;
    int numSteps = 100;

    for (int step = 0; step < numSteps; ++step) {
        updateParticles<<<blocksPerGrid, threadsPerBlock>>>(d_particles, dt);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to update particles: " << hipGetErrorString(err) << std::endl;
            hipFree(d_particles);
            hipFree(d_states);
            return -1;
        }
    }

    hipFree(d_particles);
    hipFree(d_states);

    return 0;
}