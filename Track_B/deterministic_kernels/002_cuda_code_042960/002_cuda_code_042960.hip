#include <iostream>
#include <hip/hip_runtime.h>


// Define the number of boundary elements
#define NUM_ELEMENTS 32

// Kernel to simulate fluid-structure interaction
__global__ void simulateInteraction(float* positions, float* forces, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        // Simple force application: move position based on force and time step
        positions[idx] += forces[idx] * dt;
        // Reset force for next iteration
        forces[idx] = 0.0f;
    }
}

// Kernel to compute forces based on current positions
__global__ void computeForces(float* positions, float* forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        // Simple force computation: attract to center
        float center = 0.5f;
        forces[idx] += (center - positions[idx]) * 0.1f;
    }
}

// Main function to demonstrate the usage of the kernels
int main() {
    // Host data
    float h_positions[NUM_ELEMENTS];
    float h_forces[NUM_ELEMENTS];

    // Initialize positions randomly
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        h_positions[i] = static_cast<float>(rand()) / RAND_MAX;
        h_forces[i] = 0.0f;
    }

    // Device data
    float* d_positions;
    float* d_forces;

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_positions, NUM_ELEMENTS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for positions" << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_forces, NUM_ELEMENTS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for forces" << std::endl;
        hipFree(d_positions);
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_positions, h_positions, NUM_ELEMENTS * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy positions from host to device" << std::endl;
        hipFree(d_positions);
        hipFree(d_forces);
        return -1;
    }
    err = hipMemcpy(d_forces, h_forces, NUM_ELEMENTS * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy forces from host to device" << std::endl;
        hipFree(d_positions);
        hipFree(d_forces);
        return -1;
    }

    // Define block and grid sizes
    dim3 blockSize(32);
    dim3 gridSize((NUM_ELEMENTS + blockSize.x - 1) / blockSize.x);

    // Time step
    float dt = 0.01f;

    // Launch computeForces kernel
    computeForces<<<gridSize, blockSize>>>(d_positions, d_forces);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "computeForces kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_positions);
        hipFree(d_forces);
        return -1;
    }

    // Launch simulateInteraction kernel
    simulateInteraction<<<gridSize, blockSize>>>(d_positions, d_forces, dt);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "simulateInteraction kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_positions);
        hipFree(d_forces);
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_positions, d_positions, NUM_ELEMENTS * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy positions from device to host" << std::endl;
        hipFree(d_positions);
        hipFree(d_forces);
        return -1;
    }

    // Print results
    std::cout << "Updated positions:" << std::endl;
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        std::cout << h_positions[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    hipFree(d_positions);
    hipFree(d_forces);

    return 0;
}