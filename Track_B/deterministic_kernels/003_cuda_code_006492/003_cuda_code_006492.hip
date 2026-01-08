#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 512
#define BLOCK_SIZE 128

__constant__ float kSpring = 1.0f;
__constant__ float dt = 0.01f;
__constant__ float mass = 1.0f;

struct Particle {
    float4 pos;
    float4 vel;
};

// Kernel to update positions and velocities using Velocity Verlet algorithm
__global__ void velocityVerlet(Particle* particles, float4* forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load position and velocity from global memory
    float4 pos = particles[idx].pos;
    float4 vel = particles[idx].vel;
    float4 force = forces[idx];

    // Update position
    pos.x += vel.x * dt + 0.5f * (force.x / mass) * dt * dt;
    pos.y += vel.y * dt + 0.5f * (force.y / mass) * dt * dt;
    pos.z += vel.z * dt + 0.5f * (force.z / mass) * dt * dt;

    // Store updated position
    particles[idx].pos = pos;

    // Compute acceleration
    float4 acc;
    acc.x = force.x / mass;
    acc.y = force.y / mass;
    acc.z = force.z / mass;

    // Update velocity
    vel.x += acc.x * dt;
    vel.y += acc.y * dt;
    vel.z += acc.z * dt;

    // Store updated velocity
    particles[idx].vel = vel;
}

// Kernel to compute forces between particles
__global__ void computeForces(Particle* particles, float4* forces) {
    extern __shared__ Particle sharedParticles[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load particle data into shared memory to avoid bank conflicts
    sharedParticles[threadIdx.x] = particles[idx];
    __syncthreads();

    // Initialize force
    float4 force = {0.0f, 0.0f, 0.0f, 0.0f};

    // Compute forces using a simple harmonic potential
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        if (i != threadIdx.x) {
            float4 diff;
            diff.x = sharedParticles[i].pos.x - sharedParticles[threadIdx.x].pos.x;
            diff.y = sharedParticles[i].pos.y - sharedParticles[threadIdx.x].pos.y;
            diff.z = sharedParticles[i].pos.z - sharedParticles[threadIdx.x].pos.z;

            float rSquared = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
            float r = sqrtf(rSquared);

            float f = kSpring * (1.0f - r);

            force.x += f * diff.x / r;
            force.y += f * diff.y / r;
            force.z += f * diff.z / r;
        }
    }

    // Store computed force
    forces[idx] = force;
}

// Initialize particles with random positions and velocities
__global__ void initializeParticles(Particle* particles, hiprandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Initialize curand state
    hiprandState localState = states[idx];
    hiprand_init((unsigned long long) clock() + idx, 0, 0, &localState);

    // Set random position and velocity
    particles[idx].pos.x = hiprand_uniform(&localState) * 10.0f;
    particles[idx].pos.y = hiprand_uniform(&localState) * 10.0f;
    particles[idx].pos.z = hiprand_uniform(&localState) * 10.0f;
    particles[idx].pos.w = 1.0f; // W component for padding

    particles[idx].vel.x = hiprand_uniform(&localState) * 2.0f - 1.0f;
    particles[idx].vel.y = hiprand_uniform(&localState) * 2.0f - 1.0f;
    particles[idx].vel.z = hiprand_uniform(&localState) * 2.0f - 1.0f;
    particles[idx].vel.w = 0.0f; // W component for padding

    // Store the state back to global memory
    states[idx] = localState;
}

int main() {
    Particle* d_particles;
    float4* d_forces;
    hiprandState* d_states;

    // Allocate memory on device
    hipMalloc((void**)&d_particles, N * sizeof(Particle));
    hipMalloc((void**)&d_forces, N * sizeof(float4));
    hipMalloc((void**)&d_states, N * sizeof(hiprandState));

    // Initialize particles
    initializeParticles<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles, d_states);
    hipDeviceSynchronize();

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing particles: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Simulate for a few steps
    for (int step = 0; step < 100; ++step) {
        // Compute forces
        computeForces<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(Particle)>>>(d_particles, d_forces);
        hipDeviceSynchronize();

        // Check for errors
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error computing forces: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Update positions and velocities
        velocityVerlet<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles, d_forces);
        hipDeviceSynchronize();

        // Check for errors
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error updating particles: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Free device memory
    hipFree(d_particles);
    hipFree(d_forces);
    hipFree(d_states);

    return 0;
}