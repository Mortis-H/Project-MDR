// leapfrog_md.cu
// A simple CUDA implementation of a leapfrog integrator for molecular dynamics simulation.
// Optimized for register usage on Ada Lovelace architecture.

#include <iostream>
#include <hip/hip_runtime.h>

// Constants for the simulation
#define NUM_PARTICLES 512
#define DT 0.001f
#define SIMULATION_STEPS 100

// Kernel to update positions using leapfrog method
__global__ void updatePositions(float4 *positions, float4 *velocities, float4 *forces, float dt, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        // Update position using leapfrog method
        positions[idx].x += velocities[idx].x * dt + 0.5f * forces[idx].x * dt * dt;
        positions[idx].y += velocities[idx].y * dt + 0.5f * forces[idx].y * dt * dt;
        positions[idx].z += velocities[idx].z * dt + 0.5f * forces[idx].z * dt * dt;
    }
}

// Kernel to update velocities using leapfrog method
__global__ void updateVelocities(float4 *velocities, float4 *forces, float dt, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        // Update velocity using leapfrog method
        velocities[idx].x += forces[idx].x * dt;
        velocities[idx].y += forces[idx].y * dt;
        velocities[idx].z += forces[idx].z * dt;
    }
}

// Kernel to compute forces (simple Lennard-Jones potential for demonstration)
__global__ void computeForces(float4 *positions, float4 *forces, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        float4 force = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int j = 0; j < numParticles; ++j) {
            if (idx != j) {
                float dx = positions[j].x - positions[idx].x;
                float dy = positions[j].y - positions[idx].y;
                float dz = positions[j].z - positions[idx].z;
                float r2 = dx * dx + dy * dy + dz * dz;
                float r6 = r2 * r2 * r2;
                float r12 = r6 * r6;
                float epsilon = 1.0f;
                float sigma = 1.0f;
                float f = 24.0f * epsilon * (2.0f * sigma * sigma * sigma * sigma * sigma * sigma / r12 - 1.0f / r6) / r2;
                force.x += f * dx;
                force.y += f * dy;
                force.z += f * dz;
            }
        }
        forces[idx] = force;
    }
}

int main() {
    // Allocate memory for positions, velocities, and forces
    float4 *h_positions, *h_velocities, *h_forces;
    float4 *d_positions, *d_velocities, *d_forces;
    size_t size = NUM_PARTICLES * sizeof(float4);

    h_positions = (float4 *)malloc(size);
    h_velocities = (float4 *)malloc(size);
    h_forces = (float4 *)malloc(size);

    hipMalloc((void **)&d_positions, size);
    hipMalloc((void **)&d_velocities, size);
    hipMalloc((void **)&d_forces, size);

    // Initialize positions and velocities
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_positions[i] = make_float4(i, i, i, 0.0f);
        h_velocities[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Copy data from host to device
    hipMemcpy(d_positions, h_positions, size, hipMemcpyHostToDevice);
    hipMemcpy(d_velocities, h_velocities, size, hipMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (NUM_PARTICLES + blockSize - 1) / blockSize;

    // Leapfrog integration
    for (int step = 0; step < SIMULATION_STEPS; ++step) {
        // Step 1: Update positions
        updatePositions<<<gridSize, blockSize>>>(d_positions, d_velocities, d_forces, DT, NUM_PARTICLES);
        hipDeviceSynchronize();

        // Step 2: Compute forces
        computeForces<<<gridSize, blockSize>>>(d_positions, d_forces, NUM_PARTICLES);
        hipDeviceSynchronize();

        // Step 3: Update velocities
        updateVelocities<<<gridSize, blockSize>>>(d_velocities, d_forces, DT, NUM_PARTICLES);
        hipDeviceSynchronize();
    }

    // Copy results back to host
    hipMemcpy(h_positions, d_positions, size, hipMemcpyDeviceToHost);
    hipMemcpy(h_velocities, d_velocities, size, hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_forces);

    // Free host memory
    free(h_positions);
    free(h_velocities);
    free(h_forces);

    // Check for any errors launching the kernel
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Simulation completed successfully." << std::endl;
    return 0;
}