#include <iostream>
#include <hip/hip_runtime.h>

#include <hiprand/hiprand_kernel.h>

#define NUM_PARTICLES 512
#define BLOCK_SIZE 256

// Define a simple particle structure
struct Particle {
    float4 pos; // x, y, z, w (mass)
    float4 vel; // vx, vy, vz, padding
};

// Kernel to perform N-body simulation
__global__ void nbodyKernel(Particle* particles, float dt, int numParticles) {
    extern __shared__ Particle sharedParticles[];

    // Load particles into shared memory
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numParticles) {
        sharedParticles[threadIdx.x] = particles[idx];
    }
    __syncthreads();

    // Initialize acceleration
    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // Calculate forces
    for (int i = 0; i < numParticles; i += blockDim.x) {
        int j = i + threadIdx.x;
        if (j < numParticles && j != idx) {
            float4 r = {sharedParticles[threadIdx.x].pos.x - sharedParticles[j % blockDim.x].pos.x,
                        sharedParticles[threadIdx.x].pos.y - sharedParticles[j % blockDim.x].pos.y,
                        sharedParticles[threadIdx.x].pos.z - sharedParticles[j % blockDim.x].pos.z,
                        0.0f};
            float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 1e-10f;
            float invDist = rsqrtf(distSqr);
            float invDistCube = invDist * invDist * invDist;
            float s = sharedParticles[j % blockDim.x].pos.w * invDistCube;

            acc.x += r.x * s;
            acc.y += r.y * s;
            acc.z += r.z * s;
        }
        __syncthreads();
    }

    // Update velocity and position
    if (idx < numParticles) {
        particles[idx].vel.x += acc.x * dt;
        particles[idx].vel.y += acc.y * dt;
        particles[idx].vel.z += acc.z * dt;

        particles[idx].pos.x += particles[idx].vel.x * dt;
        particles[idx].pos.y += particles[idx].vel.y * dt;
        particles[idx].pos.z += particles[idx].vel.z * dt;
    }
}

// Initialize particles with random positions and velocities
__global__ void initParticlesKernel(Particle* particles, int numParticles) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numParticles) {
        hiprandState state;
        hiprand_init((unsigned long long) clock() + idx, 0, 0, &state);

        particles[idx].pos.x = hiprand_uniform(&state) * 100.0f;
        particles[idx].pos.y = hiprand_uniform(&state) * 100.0f;
        particles[idx].pos.z = hiprand_uniform(&state) * 100.0f;
        particles[idx].pos.w = 1.0f; // mass

        particles[idx].vel.x = hiprand_uniform(&state) * 2.0f - 1.0f;
        particles[idx].vel.y = hiprand_uniform(&state) * 2.0f - 1.0f;
        particles[idx].vel.z = hiprand_uniform(&state) * 2.0f - 1.0f;
    }
}

int main() {
    // Number of particles
    int numParticles = NUM_PARTICLES;

    // Allocate memory on the host
    Particle* h_particles = (Particle*)malloc(numParticles * sizeof(Particle));

    // Allocate memory on the device
    Particle* d_particles;
    hipMalloc((void**)&d_particles, numParticles * sizeof(Particle));

    // Initialize particles on the device
    int numBlocks = (numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initParticlesKernel<<<numBlocks, BLOCK_SIZE>>>(d_particles, numParticles);
    hipDeviceSynchronize();

    // Simulation parameters
    float dt = 0.01f;
    int numSteps = 100;

    // Perform N-body simulation
    for (int step = 0; step < numSteps; step++) {
        nbodyKernel<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(Particle)>>>(d_particles, dt, numParticles);
        hipDeviceSynchronize();
    }

    // Copy results back to host
    hipMemcpy(h_particles, d_particles, numParticles * sizeof(Particle), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_particles);

    // Free host memory
    free(h_particles);

    return 0;
}