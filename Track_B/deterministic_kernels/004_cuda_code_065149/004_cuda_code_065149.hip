#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define BLOCK_SIZE 256
#define NUM_ELEMENTS 4096

__global__ void initializeBoundaryElements(float* positions, float* velocities, hiprandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        hiprandState localState = states[idx];
        hiprand_init(seed, idx, 0, &localState);
        positions[idx] = hiprand_uniform(&localState) * 10.0f; // Random position
        velocities[idx] = hiprand_uniform(&localState) * 2.0f - 1.0f; // Random velocity
        states[idx] = localState;
    }
}

__global__ void updateBoundaryElements(float* positions, float* velocities, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        // Simple update rule: position += velocity * deltaTime
        positions[idx] += velocities[idx] * deltaTime;
        // Apply boundary conditions (simple periodic boundary)
        if (positions[idx] > 10.0f) positions[idx] -= 10.0f;
        if (positions[idx] < 0.0f) positions[idx] += 10.0f;
    }
}

__global__ void fluidInteraction(float* positions, float* velocities, float* fluidForces, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        // Simple fluid interaction: velocity += force * deltaTime
        velocities[idx] += fluidForces[idx] * deltaTime;
        // Clamp velocity to prevent unphysical values
        if (velocities[idx] > 5.0f) velocities[idx] = 5.0f;
        if (velocities[idx] < -5.0f) velocities[idx] = -5.0f;
    }
}

__global__ void computeFluidForces(float* positions, float* fluidForces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        // Simple force computation: force proportional to distance from center
        float distanceFromCenter = fabsf(positions[idx] - 5.0f);
        fluidForces[idx] = -0.1f * distanceFromCenter; // Simple attractive force
    }
}

int main() {
    float* d_positions, *d_velocities, *d_fluidForces;
    hiprandState* d_states;
    float* h_positions = new float[NUM_ELEMENTS];
    float* h_velocities = new float[NUM_ELEMENTS];
    float* h_fluidForces = new float[NUM_ELEMENTS];

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_positions, NUM_ELEMENTS * sizeof(float));
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_velocities, NUM_ELEMENTS * sizeof(float));
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_fluidForces, NUM_ELEMENTS * sizeof(float));
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_states, NUM_ELEMENTS * sizeof(hiprandState));
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

    // Initialize random states
    initializeBoundaryElements<<<(NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_positions, d_velocities, d_states, 12345);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

    // Simulation loop
    const int numSteps = 100;
    const float deltaTime = 0.01f;
    for (int step = 0; step < numSteps; ++step) {
        // Compute fluid forces
        computeFluidForces<<<(NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_positions, d_fluidForces);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

        // Apply fluid interaction
        fluidInteraction<<<(NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_positions, d_velocities, d_fluidForces, deltaTime);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

        // Update boundary elements
        updateBoundaryElements<<<(NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_positions, d_velocities, deltaTime);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    }

    // Copy results back to host
    err = hipMemcpy(h_positions, d_positions, NUM_ELEMENTS * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMemcpy(h_velocities, d_velocities, NUM_ELEMENTS * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

    // Free device memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_fluidForces);
    hipFree(d_states);

    // Free host memory
    delete[] h_positions;
    delete[] h_velocities;
    delete[] h_fluidForces;

    std::cout << "Simulation completed successfully." << std::endl;
    return 0;
}