// particle_in_cell_simulation.cu
// A basic 8192D particle-in-cell fluid simulation on the Hopper architecture

#include <iostream>
#include <hip/hip_runtime.h>

#define N 8192  // Number of particles
#define BLOCK_SIZE 256

// Kernel to update particle positions and velocities
__global__ void updateParticles(float4* pos, float4* vel, float dt, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple Euler method for updating positions and velocities
        pos[idx].x += vel[idx].x * dt;
        pos[idx].y += vel[idx].y * dt;
        pos[idx].z += vel[idx].z * dt;

        // Example of a simple force (constant acceleration in z-direction)
        vel[idx].z += 9.81f * dt;
    }
}

// Kernel to deposit particle charge onto the grid
__global__ void depositCharge(float* grid, float4* pos, float charge, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Deposit charge onto the grid cell corresponding to particle position
        int gridX = static_cast<int>(pos[idx].x);
        int gridY = static_cast<int>(pos[idx].y);
        int gridIdx = gridX * N + gridY;

        // Simple charge deposition (no interpolation)
        atomicAdd(&grid[gridIdx], charge);
    }
}

int main() {
    // Initialize particle positions and velocities
    float4* h_pos = new float4[N];
    float4* h_vel = new float4[N];
    for (int i = 0; i < N; ++i) {
        h_pos[i] = make_float4(static_cast<float>(i % N), static_cast<float>(i / N), 0.0f, 0.0f);
        h_vel[i] = make_float4(0.1f, 0.1f, 0.0f, 0.0f);
    }

    // Allocate device memory
    float4* d_pos, *d_vel;
    float* d_grid;
    hipMalloc(&d_pos, N * sizeof(float4));
    hipMalloc(&d_vel, N * sizeof(float4));
    hipMalloc(&d_grid, N * N * sizeof(float));

    // Copy data to device
    hipMemcpy(d_pos, h_pos, N * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_vel, h_vel, N * sizeof(float4), hipMemcpyHostToDevice);
    hipMemset(d_grid, 0, N * N * sizeof(float));

    // Launch updateParticles kernel
    int blockSize = BLOCK_SIZE;
    int numBlocks = (N + blockSize - 1) / blockSize;
    float dt = 0.01f;
    updateParticles<<<numBlocks, blockSize>>>(d_pos, d_vel, dt, N);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error in updateParticles: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch depositCharge kernel
    float charge = 1.0f;
    depositCharge<<<numBlocks, blockSize>>>(d_grid, d_pos, charge, N);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error in depositCharge: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy grid back to host for verification (optional)
    float* h_grid = new float[N * N];
    hipMemcpy(h_grid, d_grid, N * N * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_pos);
    hipFree(d_vel);
    hipFree(d_grid);

    // Free host memory
    delete[] h_pos;
    delete[] h_vel;
    delete[] h_grid;

    std::cout << "Simulation completed successfully." << std::endl;
    return 0;
}