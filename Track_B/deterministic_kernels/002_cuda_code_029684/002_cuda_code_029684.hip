#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define BLOCK_SIZE 32
#define GRID_SIZE 32

// Structure to represent a point in 3D space
struct Point {
    float x, y, z;
};

// Kernel to simulate cloth physics
__global__ void clothSimulationKernel(Point* d_positions, float* d_forces, float deltaTime, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        // Simple Verlet integration
        Point prevPos = d_positions[index];
        d_positions[index].x += d_forces[index] * deltaTime * deltaTime;
        d_positions[index].y += d_forces[index] * deltaTime * deltaTime;
        d_positions[index].z += d_forces[index] * deltaTime * deltaTime;
        d_forces[index] = 0.0f; // Reset forces for next frame

        // Constrain the cloth to a fixed point at the top
        if (y == 0) {
            d_positions[index] = prevPos;
        }
    }
}

// Initialize random forces for each point
__global__ void initForcesKernel(float* d_forces, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        hiprandState state;
        hiprand_init((unsigned long long)clock() + index, 0, 0, &state);
        d_forces[index] = hiprand_uniform(&state) * 0.1f - 0.05f; // Random force between -0.05 and 0.05
    }
}

int main() {
    const int width = GRID_SIZE;
    const int height = GRID_SIZE;
    const int numPoints = width * height;
    const float deltaTime = 0.01f;

    // Allocate memory on the host
    Point* h_positions = new Point[numPoints];
    float* h_forces = new float[numPoints];

    // Initialize positions and forces on the host
    for (int i = 0; i < numPoints; ++i) {
        h_positions[i].x = static_cast<float>(i % width);
        h_positions[i].y = static_cast<float>(i / width);
        h_positions[i].z = 0.0f;
        h_forces[i] = 0.0f;
    }

    // Allocate memory on the device
    Point* d_positions;
    float* d_forces;
    hipError_t err = hipMalloc((void**)&d_positions, numPoints * sizeof(Point));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_positions: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_forces, numPoints * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_forces: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_positions, h_positions, numPoints * sizeof(Point), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying h_positions to d_positions: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize forces on the device
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    initForcesKernel<<<gridSize, blockSize>>>(d_forces, width, height);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error in initForcesKernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Run the cloth simulation kernel
    clothSimulationKernel<<<gridSize, blockSize>>>(d_positions, d_forces, deltaTime, width, height);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error in clothSimulationKernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy data from device to host
    err = hipMemcpy(h_positions, d_positions, numPoints * sizeof(Point), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying d_positions to h_positions: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_positions);
    hipFree(d_forces);

    // Free host memory
    delete[] h_positions;
    delete[] h_forces;

    std::cout << "Simulation completed successfully." << std::endl;
    return 0;
}