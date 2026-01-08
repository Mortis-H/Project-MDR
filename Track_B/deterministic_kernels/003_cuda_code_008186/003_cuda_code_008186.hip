#include <iostream>
#include <hip/hip_runtime.h>

// Constants for the simulation
#define NUM_PARTICLES 4096
#define GRID_SIZE 32
#define BLOCK_SIZE 128

// Kernel to update particle positions
__global__ void updateParticles(float* positions, float* velocities, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        positions[idx] += velocities[idx] * dt;
    }
}

// Kernel to update cell charges
__global__ void updateCellCharges(float* positions, float* charges, float* cellCharges) {
    extern __shared__ float sharedPositions[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    // Load particle positions into shared memory
    if (idx < NUM_PARTICLES) {
        sharedPositions[localIdx] = positions[idx];
    }
    __syncthreads();

    // Update cell charges
    for (int i = 0; i < blockDim.x; ++i) {
        if (sharedPositions[i] < (blockIdx.x + 1) * GRID_SIZE && sharedPositions[i] >= blockIdx.x * GRID_SIZE) {
            atomicAdd(&cellCharges[blockIdx.x], charges[idx]);
        }
    }
}

// Kernel to update particle velocities based on cell forces
__global__ void updateVelocities(float* positions, float* velocities, float* cellCharges, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        int cellIdx = positions[idx] / GRID_SIZE;
        float force = cellCharges[cellIdx] * 0.1f; // Simplified force calculation
        velocities[idx] += force * dt;
    }
}

// Main function to demonstrate the usage of the kernels
int main() {
    float* d_positions, *d_velocities, *d_charges, *d_cellCharges;
    float h_positions[NUM_PARTICLES], h_velocities[NUM_PARTICLES], h_charges[NUM_PARTICLES], h_cellCharges[GRID_SIZE];

    // Initialize host data
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_positions[i] = static_cast<float>(i) * 0.5f;
        h_velocities[i] = 1.0f;
        h_charges[i] = 1.0f;
    }
    for (int i = 0; i < GRID_SIZE; ++i) {
        h_cellCharges[i] = 0.0f;
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_positions, NUM_PARTICLES * sizeof(float));
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_velocities, NUM_PARTICLES * sizeof(float));
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_charges, NUM_PARTICLES * sizeof(float));
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_cellCharges, GRID_SIZE * sizeof(float));
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

    // Copy data from host to device
    err = hipMemcpy(d_positions, h_positions, NUM_PARTICLES * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMemcpy(d_velocities, h_velocities, NUM_PARTICLES * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMemcpy(d_charges, h_charges, NUM_PARTICLES * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMemcpy(d_cellCharges, h_cellCharges, GRID_SIZE * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

    // Launch kernels
    float dt = 0.1f;
    updateParticles<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_positions, d_velocities, dt);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

    updateCellCharges<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_positions, d_charges, d_cellCharges);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

    updateVelocities<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_positions, d_velocities, d_cellCharges, dt);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

    // Copy results back to host
    err = hipMemcpy(h_positions, d_positions, NUM_PARTICLES * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMemcpy(h_velocities, d_velocities, NUM_PARTICLES * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMemcpy(h_cellCharges, d_cellCharges, GRID_SIZE * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

    // Free device memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_charges);
    hipFree(d_cellCharges);

    // Output results
    std::cout << "Updated positions and velocities:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "Particle " << i << ": Position = " << h_positions[i] << ", Velocity = " << h_velocities[i] << std::endl;
    }

    return 0;
}