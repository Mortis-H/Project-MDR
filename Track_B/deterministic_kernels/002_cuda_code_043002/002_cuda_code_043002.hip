#include <iostream>
#include <hip/hip_runtime.h>

// Define the size of the grid
#define GRID_SIZE 8192

// Kernel to simulate cloth physics
__global__ void clothSimulationKernel(float* d_positions, float* d_velocities, float* d_forces, float deltaTime, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= width) return;

    int idx = y * width + x;

    // Simple Euler integration for position and velocity update
    d_positions[idx] += d_velocities[idx] * deltaTime;
    d_velocities[idx] += d_forces[idx] * deltaTime;

    // Clear forces for next iteration
    d_forces[idx] = 0.0f;
}

// Kernel to apply forces (e.g., gravity)
__global__ void applyGravityKernel(float* d_forces, int width, float gravity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= width) return;

    int idx = y * width + x;

    // Apply gravity force
    d_forces[idx] += gravity;
}

int main() {
    // Initialize CUDA device
    int device = 0;
    hipSetDevice(device);

    // Allocate memory for positions, velocities, and forces
    float* d_positions, *d_velocities, *d_forces;
    size_t size = GRID_SIZE * GRID_SIZE * sizeof(float);
    hipMalloc(&d_positions, size);
    hipMalloc(&d_velocities, size);
    hipMalloc(&d_forces, size);

    // Initialize positions, velocities, and forces to zero
    hipMemset(d_positions, 0, size);
    hipMemset(d_velocities, 0, size);
    hipMemset(d_forces, 0, size);

    // Define block size
    dim3 blockSize(16, 16);
    dim3 gridSize((GRID_SIZE + blockSize.x - 1) / blockSize.x, (GRID_SIZE + blockSize.y - 1) / blockSize.y);

    // Simulation parameters
    float deltaTime = 0.01f;
    float gravity = -9.81f;

    // Launch kernels
    for (int i = 0; i < 100; ++i) { // Simulate for 100 time steps
        // Apply gravity
        applyGravityKernel<<<gridSize, blockSize>>>(d_forces, GRID_SIZE, gravity);
        hipDeviceSynchronize();

        // Simulate cloth physics
        clothSimulationKernel<<<gridSize, blockSize>>>(d_positions, d_velocities, d_forces, deltaTime, GRID_SIZE);
        hipDeviceSynchronize();
    }

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_forces);

    std::cout << "Cloth simulation completed successfully." << std::endl;
    return 0;
}