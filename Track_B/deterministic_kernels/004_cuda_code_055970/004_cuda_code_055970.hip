#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 8192

// Kernel to update particle positions
__global__ void updateParticles(float* positions, float* velocities, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Update position using velocity
        positions[idx] += velocities[idx] * dt;
    }
}

// Kernel to deposit charge from particles to grid
__global__ void depositCharge(float* grid, float* positions, float* charges, float cellSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int gridIdx = static_cast<int>(positions[idx] / cellSize);
        if (gridIdx < N) {
            atomicAdd(&grid[gridIdx], charges[idx]);
        }
    }
}

// Kernel to initialize particles with random positions and velocities
__global__ void initializeParticles(float* positions, float* velocities, float* charges, hiprandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        hiprandState localState = states[idx];
        positions[idx] = hiprand_uniform(&localState) * N; // Random position within domain
        velocities[idx] = hiprand_uniform(&localState) * 2.0f - 1.0f; // Random velocity in [-1, 1]
        charges[idx] = 1.0f; // All particles have the same charge
        states[idx] = localState;
    }
}

// Initialize random states for curand
__global__ void initCurandStates(hiprandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        hiprand_init(seed, idx, 0, &states[idx]);
    }
}

int main() {
    float* d_positions, *d_velocities, *d_charges, *d_grid;
    hiprandState* d_states;
    float* h_positions = new float[N];
    float* h_velocities = new float[N];
    float* h_charges = new float[N];
    float* h_grid = new float[N];
    const float dt = 0.01f;
    const float cellSize = 1.0f; // Assuming domain size is N

    // Allocate device memory
    hipMalloc((void**)&d_positions, N * sizeof(float));
    hipMalloc((void**)&d_velocities, N * sizeof(float));
    hipMalloc((void**)&d_charges, N * sizeof(float));
    hipMalloc((void**)&d_grid, N * sizeof(float));
    hipMalloc((void**)&d_states, N * sizeof(hiprandState));

    // Initialize random states
    initCurandStates<<<(N + 255) / 256, 256>>>(d_states, 1234);
    hipDeviceSynchronize();

    // Initialize particles
    initializeParticles<<<(N + 255) / 256, 256>>>(d_positions, d_velocities, d_charges, d_states);
    hipDeviceSynchronize();

    // Zero out grid
    hipMemset(d_grid, 0, N * sizeof(float));

    // Update particles and deposit charge
    updateParticles<<<(N + 255) / 256, 256>>>(d_positions, d_velocities, dt);
    hipDeviceSynchronize();

    depositCharge<<<(N + 255) / 256, 256>>>(d_grid, d_positions, d_charges, cellSize);
    hipDeviceSynchronize();

    // Copy results back to host
    hipMemcpy(h_positions, d_positions, N * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_velocities, d_velocities, N * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_charges, d_charges, N * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_grid, d_grid, N * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_charges);
    hipFree(d_grid);
    hipFree(d_states);

    // Free host memory
    delete[] h_positions;
    delete[] h_velocities;
    delete[] h_charges;
    delete[] h_grid;

    return 0;
}