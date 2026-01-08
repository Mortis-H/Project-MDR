#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>

namespace cg = cooperative_groups;

#define BLOCK_SIZE 256
#define NUM_PARTICLES 128
#define GRID_SIZE (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE

// Kernel to calculate forces using particle mesh
__global__ void calculateForcesKernel(float4* positions, float4* forces, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float4 pos = positions[idx];
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Simple N-body force calculation (not using PME for brevity)
    for (int j = 0; j < numParticles; ++j) {
        if (j == idx) continue;
        float4 otherPos = positions[j];
        float4 diff = make_float4(otherPos.x - pos.x, otherPos.y - pos.y, otherPos.z - pos.z, 0.0f);
        float distSqr = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + 1e-10f;
        float dist = sqrtf(distSqr);
        float invDist3 = 1.0f / (dist * distSqr);
        force.x += diff.x * invDist3;
        force.y += diff.y * invDist3;
        force.z += diff.z * invDist3;
    }

    forces[idx] = force;
}

// Kernel to update positions based on forces
__global__ void updatePositionsKernel(float4* positions, float4* forces, float dt, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float4 pos = positions[idx];
    float4 force = forces[idx];
    pos.x += force.x * dt;
    pos.y += force.y * dt;
    pos.z += force.z * dt;
    positions[idx] = pos;
}

// Main function to demonstrate the usage of the kernels
int main() {
    // Initialize random number generator
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Initialize positions and forces on host
    thrust::host_vector<float4> h_positions(NUM_PARTICLES);
    thrust::host_vector<float4> h_forces(NUM_PARTICLES, make_float4(0.0f, 0.0f, 0.0f, 0.0f));

    // Randomly initialize positions
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_positions[i] = make_float4(dist(rng), dist(rng), dist(rng), 0.0f);
    }

    // Allocate memory on device
    float4* d_positions;
    float4* d_forces;
    hipMalloc(&d_positions, NUM_PARTICLES * sizeof(float4));
    hipMalloc(&d_forces, NUM_PARTICLES * sizeof(float4));

    // Copy data from host to device
    hipMemcpy(d_positions, h_positions.data(), NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_forces, h_forces.data(), NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);

    // Simulation parameters
    float dt = 0.01f;
    int numSteps = 100;

    // Perform N-body simulation
    for (int step = 0; step < numSteps; ++step) {
        // Launch force calculation kernel
        calculateForcesKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_positions, d_forces, NUM_PARTICLES);
        hipDeviceSynchronize();
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Kernel error: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Launch position update kernel
        updatePositionsKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_positions, d_forces, dt, NUM_PARTICLES);
        hipDeviceSynchronize();
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Kernel error: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy results back to host
    hipMemcpy(h_positions.data(), d_positions, NUM_PARTICLES * sizeof(float4), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_positions);
    hipFree(d_forces);

    // Output final positions (for demonstration purposes)
    std::cout << "Final positions of particles:" << std::endl;
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        std::cout << "Particle " << i << ": (" << h_positions[i].x << ", " << h_positions[i].y << ", " << h_positions[i].z << ")" << std::endl;
    }

    return 0;
}