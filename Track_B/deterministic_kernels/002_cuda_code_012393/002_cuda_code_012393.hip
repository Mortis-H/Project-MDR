#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cmath>

namespace cg = cooperative_groups;

// Constants
const int N = 8192;       // Number of particles
const float dt = 0.001f;  // Time step
const float epsilon = 1.0f; // Lennard-Jones epsilon
const float sigma = 1.0f; // Lennard-Jones sigma
const float rc = 2.5f;    // Cutoff radius

// Particle structure
struct Particle {
    float4 pos;  // Position (x, y, z, w=unused)
    float4 vel;  // Velocity (vx, vy, vz, w=unused)
    float4 acc;  // Acceleration (ax, ay, az, w=unused)
};

// CUDA kernel to compute forces using Lennard-Jones potential
__global__ void computeForces(Particle* particles, float4* forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float4 force = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 pos = particles[idx].pos;

    // Use cooperative groups to optimize memory access
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float4 sharedPos[256];

    for (int j = 0; j < N; j += blockDim.x) {
        int jdx = j + threadIdx.x;
        if (jdx < N) {
            sharedPos[threadIdx.x] = particles[jdx].pos;
        }
        cta.sync();

        for (int k = 0; k < blockDim.x && j + k < N; ++k) {
            float4 otherPos = sharedPos[k];
            float dx = pos.x - otherPos.x;
            float dy = pos.y - otherPos.y;
            float dz = pos.z - otherPos.z;
            float r2 = dx * dx + dy * dy + dz * dz;

            if (r2 < rc * rc && r2 > 0.0f) {
                float r6 = r2 * r2 * r2;
                float r12 = r6 * r6;
                float fmag = 24.0f * epsilon * ((2.0f * sigma * sigma * sigma * sigma * sigma * sigma / r12) - (sigma * sigma * sigma * sigma / r6)) / r2;

                force.x += fmag * dx;
                force.y += fmag * dy;
                force.z += fmag * dz;
            }
        }
        cta.sync();
    }

    forces[idx] = force;
}

// CUDA kernel to update positions and velocities using Velocity Verlet
__global__ void velocityVerlet(Particle* particles, float4* forces, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float4 pos = particles[idx].pos;
    float4 vel = particles[idx].vel;
    float4 acc = particles[idx].acc;
    float4 force = forces[idx];

    // Update position
    pos.x += vel.x * dt + 0.5f * acc.x * dt * dt;
    pos.y += vel.y * dt + 0.5f * acc.y * dt * dt;
    pos.z += vel.z * dt + 0.5f * acc.z * dt * dt;

    // Update velocity
    float4 newAcc = {force.x, force.y, force.z, 0.0f};
    vel.x += 0.5f * (acc.x + newAcc.x) * dt;
    vel.y += 0.5f * (acc.y + newAcc.y) * dt;
    vel.z += 0.5f * (acc.z + newAcc.z) * dt;

    // Update particle
    particles[idx].pos = pos;
    particles[idx].vel = vel;
    particles[idx].acc = newAcc;
}

int main() {
    // Allocate unified memory for particles and forces
    Particle* particles;
    float4* forces;
    hipMallocManaged(&particles, N * sizeof(Particle));
    hipMallocManaged(&forces, N * sizeof(float4));

    // Initialize particles with random positions and zero velocities
    for (int i = 0; i < N; ++i) {
        particles[i].pos = {static_cast<float>(rand()) / RAND_MAX * 10.0f,
                            static_cast<float>(rand()) / RAND_MAX * 10.0f,
                            static_cast<float>(rand()) / RAND_MAX * 10.0f,
                            0.0f};
        particles[i].vel = {0.0f, 0.0f, 0.0f, 0.0f};
        particles[i].acc = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    // Set grid and block sizes
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Perform a few steps of molecular dynamics
    for (int step = 0; step < 100; ++step) {
        // Compute forces
        computeForces<<<gridSize, blockSize>>>(particles, forces);
        hipDeviceSynchronize();

        // Update positions and velocities
        velocityVerlet<<<gridSize, blockSize>>>(particles, forces, dt);
        hipDeviceSynchronize();
    }

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free memory
    hipFree(particles);
    hipFree(forces);

    return 0;
}