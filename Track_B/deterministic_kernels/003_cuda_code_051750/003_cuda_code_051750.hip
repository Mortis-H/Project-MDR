#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define constants
#define GRID_SIZE 4096
#define BLOCK_SIZE 32
#define NUM_THREADS (GRID_SIZE * GRID_SIZE)

// Kernel to simulate cloth movement
__global__ void clothSimulationKernel(float* positions, float* velocities, float* forces, float dt, int width, int height) {
    // Calculate global thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if thread is within bounds
    if (x < width && y < height) {
        int idx = y * width + x;

        // Simple Verlet integration
        float new_x = positions[idx] + velocities[idx] * dt + 0.5f * forces[idx] * dt * dt;
        float new_v = velocities[idx] + 0.5f * (forces[idx] + 0.5f * forces[idx]) * dt;

        // Update positions and velocities
        positions[idx] = new_x;
        velocities[idx] = new_v;

        // Reset forces for next iteration
        forces[idx] = 0.0f;
    }
}

// Function to initialize random forces using CURAND
__global__ void initializeForcesKernel(hiprandState* states, float* forces, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        hiprandState localState = states[idx];
        float randomForce = hiprand_uniform(&localState) * 0.1f - 0.05f; // Random force between -0.05 and 0.05
        forces[idx] = randomForce;
        states[idx] = localState;
    }
}

// Function to initialize CURAND states
__global__ void initCurandStates(hiprandState* states, unsigned long seed, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        hiprandState localState;
        hiprand_init(seed, idx, 0, &localState);
        states[idx] = localState;
    }
}

int main() {
    // Allocate memory for positions, velocities, and forces
    float* d_positions, *d_velocities, *d_forces;
    hiprandState* d_states;
    size_t size = NUM_THREADS * sizeof(float);
    size_t states_size = NUM_THREADS * sizeof(hiprandState);

    hipError_t err = hipMalloc((void**)&d_positions, size);
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for positions: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_velocities, size);
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for velocities: " << hipGetErrorString(err) << std::endl;
        hipFree(d_positions);
        return -1;
    }

    err = hipMalloc((void**)&d_forces, size);
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for forces: " << hipGetErrorString(err) << std::endl;
        hipFree(d_positions);
        hipFree(d_velocities);
        return -1;
    }

    err = hipMalloc((void**)&d_states, states_size);
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for CURAND states: " << hipGetErrorString(err) << std::endl;
        hipFree(d_positions);
        hipFree(d_velocities);
        hipFree(d_forces);
        return -1;
    }

    // Initialize CURAND states
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(GRID_SIZE / BLOCK_SIZE, GRID_SIZE / BLOCK_SIZE);
    initCurandStates<<<numBlocks, threadsPerBlock>>>(d_states, time(NULL), GRID_SIZE, GRID_SIZE);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing CURAND states: " << hipGetErrorString(err) << std::endl;
        hipFree(d_positions);
        hipFree(d_velocities);
        hipFree(d_forces);
        hipFree(d_states);
        return -1;
    }

    // Initialize positions and velocities to zero
    hipMemset(d_positions, 0, size);
    hipMemset(d_velocities, 0, size);

    // Initialize forces with random values
    initializeForcesKernel<<<numBlocks, threadsPerBlock>>>(d_states, d_forces, GRID_SIZE, GRID_SIZE);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing forces: " << hipGetErrorString(err) << std::endl;
        hipFree(d_positions);
        hipFree(d_velocities);
        hipFree(d_forces);
        hipFree(d_states);
        return -1;
    }

    // Simulation parameters
    float dt = 0.01f;
    int numSteps = 100;

    // Run the simulation
    for (int step = 0; step < numSteps; ++step) {
        // Update forces (for simplicity, reinitialize forces each step)
        initializeForcesKernel<<<numBlocks, threadsPerBlock>>>(d_states, d_forces, GRID_SIZE, GRID_SIZE);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error updating forces: " << hipGetErrorString(err) << std::endl;
            hipFree(d_positions);
            hipFree(d_velocities);
            hipFree(d_forces);
            hipFree(d_states);
            return -1;
        }

        // Run the simulation kernel
        clothSimulationKernel<<<numBlocks, threadsPerBlock>>>(d_positions, d_velocities, d_forces, dt, GRID_SIZE, GRID_SIZE);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error running simulation kernel: " << hipGetErrorString(err) << std::endl;
            hipFree(d_positions);
            hipFree(d_velocities);
            hipFree(d_forces);
            hipFree(d_states);
            return -1;
        }
    }

    // Free memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_forces);
    hipFree(d_states);

    std::cout << "Simulation completed successfully." << std::endl;

    return 0;
}