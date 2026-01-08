#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N_PARTICLES 8192
#define BLOCK_SIZE 256
#define NUM_ITERATIONS 100
#define DT 0.001f
#define EPSILON 1e-6f

// Define a simple 3D vector structure
struct Vec3 {
    float x, y, z;
};

// Initialize random positions and velocities
__global__ void initParticles(Vec3 *positions, Vec3 *velocities, hiprandState *states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_PARTICLES) {
        hiprandState localState = states[idx];
        hiprand_init(seed, idx, 0, &localState);
        positions[idx] = {hiprand_uniform(&localState) * 10.0f, hiprand_uniform(&localState) * 10.0f, hiprand_uniform(&localState) * 10.0f};
        velocities[idx] = {hiprand_uniform(&localState) * 1.0f - 0.5f, hiprand_uniform(&localState) * 1.0f - 0.5f, hiprand_uniform(&localState) * 1.0f - 0.5f};
        states[idx] = localState;
    }
}

// Leapfrog integration kernel
__global__ void leapfrogStep(Vec3 *positions, Vec3 *velocities, Vec3 *forces, float dt, int iteration) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_PARTICLES) {
        // Update positions
        positions[idx].x += velocities[idx].x * dt + 0.5f * forces[idx].x * dt * dt;
        positions[idx].y += velocities[idx].y * dt + 0.5f * forces[idx].y * dt * dt;
        positions[idx].z += velocities[idx].z * dt + 0.5f * forces[idx].z * dt * dt;

        // Update velocities (half step)
        if (iteration > 0) {
            velocities[idx].x += 0.5f * forces[idx].x * dt;
            velocities[idx].y += 0.5f * forces[idx].y * dt;
            velocities[idx].z += 0.5f * forces[idx].z * dt;
        }
    }
}

// Compute forces between particles using a simple Lennard-Jones potential
__global__ void computeForces(Vec3 *positions, Vec3 *forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_PARTICLES) {
        Vec3 f = {0.0f, 0.0f, 0.0f};
        for (int j = 0; j < N_PARTICLES; ++j) {
            if (idx != j) {
                float dx = positions[j].x - positions[idx].x;
                float dy = positions[j].y - positions[idx].y;
                float dz = positions[j].z - positions[idx].z;
                float r2 = dx * dx + dy * dy + dz * dz + EPSILON;
                float r6 = r2 * r2 * r2;
                float fmag = 48.0f * (1.0f / r6 - 0.5f / (r6 * r6)) / r2;
                f.x += fmag * dx;
                f.y += fmag * dy;
                f.z += fmag * dz;
            }
        }
        forces[idx] = f;
    }
}

int main() {
    Vec3 *positions, *velocities, *forces;
    hiprandState *states;

    // Allocate Unified Memory -- accessible from CPU or GPU
    hipError_t err = hipMallocManaged(&positions, N_PARTICLES * sizeof(Vec3));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate managed memory for positions" << std::endl;
        return -1;
    }
    err = hipMallocManaged(&velocities, N_PARTICLES * sizeof(Vec3));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate managed memory for velocities" << std::endl;
        hipFree(positions);
        return -1;
    }
    err = hipMallocManaged(&forces, N_PARTICLES * sizeof(Vec3));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate managed memory for forces" << std::endl;
        hipFree(positions);
        hipFree(velocities);
        return -1;
    }
    err = hipMallocManaged(&states, N_PARTICLES * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate managed memory for states" << std::endl;
        hipFree(positions);
        hipFree(velocities);
        hipFree(forces);
        return -1;
    }

    // Initialize particles
    int numBlocks = (N_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initParticles<<<numBlocks, BLOCK_SIZE>>>(positions, velocities, states, 1234);
    hipDeviceSynchronize();

    // Main simulation loop
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        computeForces<<<numBlocks, BLOCK_SIZE>>>(positions, forces);
        hipDeviceSynchronize();
        leapfrogStep<<<numBlocks, BLOCK_SIZE>>>(positions, velocities, forces, DT, i);
        hipDeviceSynchronize();
    }

    // Free Unified Memory
    hipFree(positions);
    hipFree(velocities);
    hipFree(forces);
    hipFree(states);

    return 0;
}