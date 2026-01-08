#include <iostream>
#include <hip/hip_runtime.h>

// Define grid dimensions
#define GRID_WIDTH 2048
#define GRID_HEIGHT 2048

// Kernel to simulate cloth physics using basic forces
__global__ void clothSimulationKernel(float* positions, float* velocities, float* forces, float deltaTime, float damping) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if thread is within the grid bounds
    if (x < GRID_WIDTH && y < GRID_HEIGHT) {
        int idx = y * GRID_WIDTH + x;

        // Update velocity with forces
        velocities[idx] += forces[idx] * deltaTime;
        velocities[idx] *= damping; // Apply damping

        // Update position with velocity
        positions[idx] += velocities[idx] * deltaTime;

        // Reset forces for next iteration
        forces[idx] = 0.0f;
    }
}

// Function to add a force to a specific point in the grid using atomicAdd for thread safety
__global__ void addForceKernel(float* forces, int x, int y, float force) {
    if (x >= 0 && x < GRID_WIDTH && y >= 0 && y < GRID_HEIGHT) {
        int idx = y * GRID_WIDTH + x;
        atomicAdd(&forces[idx], force);
    }
}

int main() {
    // Allocate memory for positions, velocities, and forces
    float* d_positions, *d_velocities, *d_forces;
    size_t size = GRID_WIDTH * GRID_HEIGHT * sizeof(float);

    hipError_t err = hipMalloc(&d_positions, size);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for positions: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc(&d_velocities, size);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for velocities: " << hipGetErrorString(err) << std::endl;
        hipFree(d_positions);
        return -1;
    }

    err = hipMalloc(&d_forces, size);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for forces: " << hipGetErrorString(err) << std::endl;
        hipFree(d_positions);
        hipFree(d_velocities);
        return -1;
    }

    // Initialize positions, velocities, and forces to zero
    hipMemset(d_positions, 0, size);
    hipMemset(d_velocities, 0, size);
    hipMemset(d_forces, 0, size);

    // Define block size
    dim3 blockSize(16, 16);
    dim3 gridSize((GRID_WIDTH + blockSize.x - 1) / blockSize.x, (GRID_HEIGHT + blockSize.y - 1) / blockSize.y);

    // Simulation parameters
    float deltaTime = 0.01f;
    float damping = 0.99f;

    // Simulate cloth for a number of iterations
    int numIterations = 100;
    for (int i = 0; i < numIterations; ++i) {
        // Apply some force to the cloth (e.g., gravity)
        float gravityForce = -9.81f * deltaTime;
        addForceKernel<<<gridSize, blockSize>>>(d_forces, GRID_WIDTH / 2, GRID_HEIGHT / 2, gravityForce);
        hipDeviceSynchronize();

        // Perform cloth simulation
        clothSimulationKernel<<<gridSize, blockSize>>>(d_positions, d_velocities, d_forces, deltaTime, damping);
        hipDeviceSynchronize();

        // Check for errors in kernel execution
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Kernel execution failed: " << hipGetErrorString(err) << std::endl;
            break;
        }
    }

    // Free allocated memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_forces);

    return 0;
}