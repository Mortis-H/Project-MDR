#include <iostream>
#include <hip/hip_runtime.h>


// Define the number of boundary elements
#define NUM_ELEMENTS 1024

// Kernel to simulate fluid-structure interaction
__global__ void fluidStructureInteraction(float* boundaryElements, float* forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        // Simulate a simple force calculation
        forces[idx] = boundaryElements[idx] * 0.1f; // Example force calculation
    }
}

// Kernel to apply forces to boundary elements
__global__ void applyForces(float* boundaryElements, float* forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        // Apply force to boundary element
        boundaryElements[idx] += forces[idx];
    }
}

int main() {
    // Host data
    float* h_boundaryElements = new float[NUM_ELEMENTS];
    float* h_forces = new float[NUM_ELEMENTS];

    // Initialize host data
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        h_boundaryElements[i] = static_cast<float>(i) / NUM_ELEMENTS;
    }

    // Device data
    float* d_boundaryElements;
    float* d_forces;

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_boundaryElements, NUM_ELEMENTS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for boundaryElements: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_forces, NUM_ELEMENTS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for forces: " << hipGetErrorString(err) << std::endl;
        hipFree(d_boundaryElements);
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_boundaryElements, h_boundaryElements, NUM_ELEMENTS * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying boundaryElements to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_boundaryElements);
        hipFree(d_forces);
        return -1;
    }

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (NUM_ELEMENTS + blockSize - 1) / blockSize;

    // Launch kernel to calculate forces
    fluidStructureInteraction<<<gridSize, blockSize>>>(d_boundaryElements, d_forces);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching fluidStructureInteraction kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_boundaryElements);
        hipFree(d_forces);
        return -1;
    }

    // Launch kernel to apply forces
    applyForces<<<gridSize, blockSize>>>(d_boundaryElements, d_forces);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching applyForces kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_boundaryElements);
        hipFree(d_forces);
        return -1;
    }

    // Copy data from device to host
    err = hipMemcpy(h_boundaryElements, d_boundaryElements, NUM_ELEMENTS * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying boundaryElements from device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_boundaryElements);
        hipFree(d_forces);
        return -1;
    }

    // Free device memory
    hipFree(d_boundaryElements);
    hipFree(d_forces);

    // Free host memory
    delete[] h_boundaryElements;
    delete[] h_forces;

    std::cout << "Fluid-Structure Interaction simulation completed successfully." << std::endl;

    return 0;
}