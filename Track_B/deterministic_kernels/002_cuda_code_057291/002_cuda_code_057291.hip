#include <iostream>
#include <hip/hip_runtime.h>


// Constants
#define GRID_SIZE 2048
#define BLOCK_SIZE 32
#define NUM_THREADS (GRID_SIZE * GRID_SIZE)

// Kernel to simulate cloth physics
__global__ void clothSimulationKernel(float* positions, float* velocities, float* forces, float dt, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // Simple Euler integration for demonstration
    velocities[idx] += forces[idx] * dt;
    positions[idx] += velocities[idx] * dt;

    // Reset forces for next iteration
    forces[idx] = 0.0f;
}

// Kernel to compute forces (simplified for demonstration)
__global__ void computeForcesKernel(float* positions, float* forces, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // Dummy force calculation
    forces[idx] = -positions[idx] * 0.1f;
}

int main() {
    // Initialize CUDA device
    int device = 0;
    hipSetDevice(device);

    // Allocate memory on host
    float* h_positions = new float[NUM_THREADS];
    float* h_velocities = new float[NUM_THREADS];
    float* h_forces = new float[NUM_THREADS];

    // Initialize host data
    for (int i = 0; i < NUM_THREADS; ++i) {
        h_positions[i] = static_cast<float>(i);
        h_velocities[i] = 0.0f;
        h_forces[i] = 0.0f;
    }

    // Allocate memory on device
    float* d_positions, *d_velocities, *d_forces;
    hipMalloc(&d_positions, NUM_THREADS * sizeof(float));
    hipMalloc(&d_velocities, NUM_THREADS * sizeof(float));
    hipMalloc(&d_forces, NUM_THREADS * sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_positions, h_positions, NUM_THREADS * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_velocities, h_velocities, NUM_THREADS * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_forces, h_forces, NUM_THREADS * sizeof(float), hipMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((NUM_THREADS + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Simulation parameters
    float dt = 0.01f;
    int numIterations = 10;

    // Perform simulation iterations
    for (int iter = 0; iter < numIterations; ++iter) {
        // Compute forces
        computeForcesKernel<<<gridSize, blockSize>>>(d_positions, d_forces, NUM_THREADS);
        hipDeviceSynchronize();

        // Simulate cloth
        clothSimulationKernel<<<gridSize, blockSize>>>(d_positions, d_velocities, d_forces, dt, NUM_THREADS);
        hipDeviceSynchronize();
    }

    // Copy results back to host
    hipMemcpy(h_positions, d_positions, NUM_THREADS * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_velocities, d_velocities, NUM_THREADS * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_forces, d_forces, NUM_THREADS * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_forces);

    // Free host memory
    delete[] h_positions;
    delete[] h_velocities;
    delete[] h_forces;

    // Check for any errors launching the kernel
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Simulation completed successfully." << std::endl;
    return 0;
}