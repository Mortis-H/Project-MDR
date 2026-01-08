#include <iostream>
#include <hip/hip_runtime.h>

// Constants for the simulation
#define NUM_PARTICLES 2048
#define DT 0.001f
#define G 0.0000000000667430f // Gravitational constant

// Structure to hold particle data
struct Particle {
    float4 pos; // x, y, z, w (w is unused)
    float4 vel; // vx, vy, vz, vw (vw is unused)
};

// CUDA kernel to compute forces between particles using a simple gravitational model
__global__ void computeForces(Particle* particles, float4* forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 pi = particles[idx].pos;

    for (int j = 0; j < NUM_PARTICLES; j++) {
        if (idx == j) continue;

        float4 pj = particles[j].pos;
        float dx = pj.x - pi.x;
        float dy = pj.y - pi.y;
        float dz = pj.z - pi.z;
        float distSqr = dx * dx + dy * dy + dz * dz;
        float dist = sqrtf(distSqr) + 1e-5f; // Avoid division by zero and numerical instability
        float invDistCube = 1.0f / (distSqr * dist);
        float f = G * invDistCube;

        force.x += f * dx;
        force.y += f * dy;
        force.z += f * dz;
    }

    forces[idx] = force;
}

// CUDA kernel to update particle positions and velocities using leapfrog integration
__global__ void updateParticles(Particle* particles, float4* forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    float4 pos = particles[idx].pos;
    float4 vel = particles[idx].vel;
    float4 force = forces[idx];

    // Update velocity
    vel.x += force.x * DT;
    vel.y += force.y * DT;
    vel.z += force.z * DT;

    // Update position
    pos.x += vel.x * DT;
    pos.y += vel.y * DT;
    pos.z += vel.z * DT;

    particles[idx].pos = pos;
    particles[idx].vel = vel;
}

int main() {
    // Allocate memory for particles and forces using CUDA Unified Memory
    Particle* particles;
    float4* forces;
    hipMallocManaged(&particles, NUM_PARTICLES * sizeof(Particle));
    hipMallocManaged(&forces, NUM_PARTICLES * sizeof(float4));

    // Initialize particles with random positions and zero velocities
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].pos = make_float4(static_cast<float>(rand()) / RAND_MAX * 100.0f,
                                        static_cast<float>(rand()) / RAND_MAX * 100.0f,
                                        static_cast<float>(rand()) / RAND_MAX * 100.0f,
                                        0.0f);
        particles[i].vel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Set up execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;

    // Simulate for a number of steps
    int numSteps = 1000;
    for (int step = 0; step < numSteps; step++) {
        // Compute forces
        computeForces<<<blocksPerGrid, threadsPerBlock>>>(particles, forces);
        hipDeviceSynchronize();
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error in computeForces: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Update particles
        updateParticles<<<blocksPerGrid, threadsPerBlock>>>(particles, forces);
        hipDeviceSynchronize();
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error in updateParticles: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Free memory
    hipFree(particles);
    hipFree(forces);

    return 0;
}