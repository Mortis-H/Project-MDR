#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 1024
#define BLOCK_SIZE 256
#define DT 0.001f
#define EPSILON 1e-10f

// Structure to represent a particle
struct Particle {
    float4 pos;
    float4 vel;
    float4 acc;
};

// Kernel to compute forces using Verlet integration
__global__ void verletIntegration(Particle* particles, float4* forces, float dt, int n) {
    extern __shared__ Particle sharedParticles[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * blockDim.x + tid;

    if (i < n) {
        // Load data into shared memory
        sharedParticles[tid] = particles[i];
        __syncthreads();

        // Compute forces using pairwise interaction
        float4 f = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int j = 0; j < blockDim.x; ++j) {
            if (j != tid) {
                float4 diff = {
                    sharedParticles[j].pos.x - sharedParticles[tid].pos.x,
                    sharedParticles[j].pos.y - sharedParticles[tid].pos.y,
                    sharedParticles[j].pos.z - sharedParticles[tid].pos.z,
                    0.0f
                };
                float r2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + EPSILON;
                float r = rsqrtf(r2);
                float fmag = 24.0f * (2.0f * powf(r, -14) - powf(r, -8));
                f.x += fmag * diff.x * r;
                f.y += fmag * diff.y * r;
                f.z += fmag * diff.z * r;
            }
        }
        __syncthreads();

        // Store computed force in global memory
        forces[i] = f;
        __syncthreads();

        // Update position and velocity using Verlet integration
        float4 newAcc = forces[i];
        float4 oldAcc = sharedParticles[tid].acc;
        float4 newPos = {
            sharedParticles[tid].pos.x + sharedParticles[tid].vel.x * dt + 0.5f * oldAcc.x * dt * dt,
            sharedParticles[tid].pos.y + sharedParticles[tid].vel.y * dt + 0.5f * oldAcc.y * dt * dt,
            sharedParticles[tid].pos.z + sharedParticles[tid].vel.z * dt + 0.5f * oldAcc.z * dt * dt,
            0.0f
        };
        float4 newVel = {
            sharedParticles[tid].vel.x + 0.5f * (oldAcc.x + newAcc.x) * dt,
            sharedParticles[tid].vel.y + 0.5f * (oldAcc.y + newAcc.y) * dt,
            sharedParticles[tid].vel.z + 0.5f * (oldAcc.z + newAcc.z) * dt,
            0.0f
        };

        // Store updated position, velocity, and acceleration
        particles[i].pos = newPos;
        particles[i].vel = newVel;
        particles[i].acc = newAcc;
    }
}

// Initialize particles with random positions and velocities
__global__ void initializeParticles(Particle* particles, hiprandState* states, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        hiprandState localState = states[idx];
        float x = hiprand_uniform(&localState) * 10.0f;
        float y = hiprand_uniform(&localState) * 10.0f;
        float z = hiprand_uniform(&localState) * 10.0f;
        float vx = hiprand_uniform(&localState) * 0.1f - 0.05f;
        float vy = hiprand_uniform(&localState) * 0.1f - 0.05f;
        float vz = hiprand_uniform(&localState) * 0.1f - 0.05f;
        particles[idx].pos = {x, y, z, 0.0f};
        particles[idx].vel = {vx, vy, vz, 0.0f};
        particles[idx].acc = {0.0f, 0.0f, 0.0f, 0.0f};
        states[idx] = localState;
    }
}

// Initialize random states
__global__ void initCurandStates(hiprandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        hiprand_init(seed, idx, 0, &states[idx]);
    }
}

int main() {
    Particle* d_particles;
    float4* d_forces;
    hiprandState* d_states;

    // Allocate memory on the device
    hipMalloc((void**)&d_particles, N * sizeof(Particle));
    hipMalloc((void**)&d_forces, N * sizeof(float4));
    hipMalloc((void**)&d_states, N * sizeof(hiprandState));

    // Initialize random states
    initCurandStates<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, 1234, N);
    hipDeviceSynchronize();

    // Initialize particles
    initializeParticles<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles, d_states, N);
    hipDeviceSynchronize();

    // Perform Verlet integration
    verletIntegration<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(Particle)>>>(d_particles, d_forces, DT, N);
    hipDeviceSynchronize();

    // Free device memory
    hipFree(d_particles);
    hipFree(d_forces);
    hipFree(d_states);

    // Check for any errors launching the kernel
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    return 0;
}