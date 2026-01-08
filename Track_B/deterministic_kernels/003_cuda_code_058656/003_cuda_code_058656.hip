#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>

#define BLOCK_SIZE 32
#define NUM_PARTICLES 1024
#define GRID_SIZE (NUM_PARTICLES / BLOCK_SIZE)

// Kernel to update particle positions
__global__ void updateParticlePositions(float* positions, float* velocities, float dt, int numParticles) {
    extern __shared__ float sharedData[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // Load particle data into shared memory
    sharedData[threadIdx.x] = positions[idx];
    __syncthreads();

    // Update particle position
    positions[idx] += velocities[idx] * dt;
}

// Kernel to deposit charge from particles to grid cells
__global__ void depositCharge(float* grid, float* positions, float* charges, int numParticles, int gridSize) {
    extern __shared__ float sharedCharges[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // Load particle charge into shared memory
    sharedCharges[threadIdx.x] = charges[idx];
    __syncthreads();

    // Deposit charge to grid cells
    int cellIdx = static_cast<int>(positions[idx]) % gridSize;
    atomicAdd(&grid[cellIdx], sharedCharges[threadIdx.x]);
}

// Kernel to solve Poisson equation on grid cells
__global__ void solvePoisson(float* grid, int gridSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= gridSize) return;

    // Simple relaxation step for demonstration
    if (idx > 0 && idx < gridSize - 1) {
        grid[idx] = 0.25f * (grid[idx - 1] + grid[idx + 1] + 2.0f * grid[idx]);
    }
}

int main() {
    // Initialize device
    hipSetDevice(0);

    // Allocate memory for particles and grid
    float* d_positions, *d_velocities, *d_charges, *d_grid;
    hipMalloc(&d_positions, NUM_PARTICLES * sizeof(float));
    hipMalloc(&d_velocities, NUM_PARTICLES * sizeof(float));
    hipMalloc(&d_charges, NUM_PARTICLES * sizeof(float));
    hipMalloc(&d_grid, GRID_SIZE * sizeof(float));

    // Initialize host data
    std::vector<float> h_positions(NUM_PARTICLES);
    std::vector<float> h_velocities(NUM_PARTICLES);
    std::vector<float> h_charges(NUM_PARTICLES);
    std::vector<float> h_grid(GRID_SIZE, 0.0f);

    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_positions[i] = static_cast<float>(i) / 10.0f;
        h_velocities[i] = 1.0f;
        h_charges[i] = 1.0f;
    }

    // Copy data to device
    hipMemcpy(d_positions, h_positions.data(), NUM_PARTICLES * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_velocities, h_velocities.data(), NUM_PARTICLES * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_charges, h_charges.data(), NUM_PARTICLES * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_grid, h_grid.data(), GRID_SIZE * sizeof(float), hipMemcpyHostToDevice);

    // Time step size
    float dt = 0.1f;

    // Perform particle-in-cell simulation
    for (int step = 0; step < 10; ++step) {
        // Update particle positions
        updateParticlePositions<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_positions, d_velocities, dt, NUM_PARTICLES);
        hipDeviceSynchronize();
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in updateParticlePositions: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Deposit charge from particles to grid cells
        depositCharge<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_grid, d_positions, d_charges, NUM_PARTICLES, GRID_SIZE);
        hipDeviceSynchronize();
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in depositCharge: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Solve Poisson equation on grid cells
        solvePoisson<<<GRID_SIZE, BLOCK_SIZE>>>(d_grid, GRID_SIZE);
        hipDeviceSynchronize();
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in solvePoisson: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy results back to host
    hipMemcpy(h_grid.data(), d_grid, GRID_SIZE * sizeof(float), hipMemcpyDeviceToHost);

    // Print grid results
    for (int i = 0; i < GRID_SIZE; ++i) {
        std::cout << "Grid cell " << i << ": " << h_grid[i] << std::endl;
    }

    // Free device memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_charges);
    hipFree(d_grid);

    return 0;
}