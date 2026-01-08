#include <iostream>
#include <hip/hip_runtime.h>


// Define grid and block dimensions
#define GRID_SIZE 16384
#define BLOCK_SIZE 256
#define TOTAL_THREADS (GRID_SIZE * GRID_SIZE)

// Kernel to simulate cloth physics
__global__ void clothSimulationKernel(float* positions, float* velocities, float* forces, float deltaTime, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * width + idx;

    if (idx < width && idy < width) {
        // Simple Verlet integration
        float prevPos = positions[index];
        positions[index] += velocities[index] * deltaTime + 0.5f * forces[index] * deltaTime * deltaTime;
        velocities[index] += 0.5f * forces[index] * deltaTime;

        // Reset force for next iteration
        forces[index] = 0.0f;

        // Boundary conditions
        if (idx == 0 || idx == width - 1 || idy == 0 || idy == width - 1) {
            positions[index] = prevPos; // Pin boundary points
        }
    }
}

// Dynamic parallelism kernel to update forces
__global__ void updateForcesKernel(float* positions, float* forces, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * width + idx;

    if (idx < width && idy < width) {
        // Simple force calculation (Hooke's law)
        float k = 100.0f; // Spring constant
        float restLength = 1.0f; // Rest length of spring
        float force = 0.0f;

        // Check left neighbor
        if (idx > 0) {
            float distance = positions[index] - positions[index - 1];
            force += -k * (distance - restLength);
        }

        // Check right neighbor
        if (idx < width - 1) {
            float distance = positions[index + 1] - positions[index];
            force += -k * (distance - restLength);
        }

        // Check bottom neighbor
        if (idy > 0) {
            float distance = positions[index] - positions[index - width];
            force += -k * (distance - restLength);
        }

        // Check top neighbor
        if (idy < width - 1) {
            float distance = positions[index + width] - positions[index];
            force += -k * (distance - restLength);
        }

        forces[index] += force;
    }
}

int main() {
    // Allocate memory for positions, velocities, and forces
    float* d_positions, *d_velocities, *d_forces;
    size_t size = TOTAL_THREADS * sizeof(float);

    hipError_t err = hipMalloc(&d_positions, size);
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for positions: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc(&d_velocities, size);
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for velocities: " << hipGetErrorString(err) << std::endl;
        hipFree(d_positions);
        return -1;
    }

    err = hipMalloc(&d_forces, size);
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for forces: " << hipGetErrorString(err) << std::endl;
        hipFree(d_positions);
        hipFree(d_velocities);
        return -1;
    }

    // Initialize positions, velocities, and forces
    float* h_positions = new float[TOTAL_THREADS];
    float* h_velocities = new float[TOTAL_THREADS];
    float* h_forces = new float[TOTAL_THREADS];

    for (int i = 0; i < TOTAL_THREADS; ++i) {
        h_positions[i] = 0.0f;
        h_velocities[i] = 0.0f;
        h_forces[i] = 0.0f;
    }

    err = hipMemcpy(d_positions, h_positions, size, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying positions to device: " << hipGetErrorString(err) << std::endl;
        delete[] h_positions;
        delete[] h_velocities;
        delete[] h_forces;
        hipFree(d_positions);
        hipFree(d_velocities);
        hipFree(d_forces);
        return -1;
    }

    err = hipMemcpy(d_velocities, h_velocities, size, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying velocities to device: " << hipGetErrorString(err) << std::endl;
        delete[] h_positions;
        delete[] h_velocities;
        delete[] h_forces;
        hipFree(d_positions);
        hipFree(d_velocities);
        hipFree(d_forces);
        return -1;
    }

    err = hipMemcpy(d_forces, h_forces, size, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying forces to device: " << hipGetErrorString(err) << std::endl;
        delete[] h_positions;
        delete[] h_velocities;
        delete[] h_forces;
        hipFree(d_positions);
        hipFree(d_velocities);
        hipFree(d_forces);
        return -1;
    }

    delete[] h_positions;
    delete[] h_velocities;
    delete[] h_forces;

    // Set up block and grid dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(GRID_SIZE / BLOCK_SIZE, GRID_SIZE / BLOCK_SIZE);

    // Simulation parameters
    float deltaTime = 0.01f;
    int numSteps = 10;

    // Perform cloth simulation
    for (int step = 0; step < numSteps; ++step) {
        // Launch update forces kernel using dynamic parallelism
        updateForcesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_positions, d_forces, GRID_SIZE);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error launching updateForcesKernel: " << hipGetErrorString(err) << std::endl;
            hipFree(d_positions);
            hipFree(d_velocities);
            hipFree(d_forces);
            return -1;
        }

        // Launch cloth simulation kernel
        clothSimulationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_positions, d_velocities, d_forces, deltaTime, GRID_SIZE);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error launching clothSimulationKernel: " << hipGetErrorString(err) << std::endl;
            hipFree(d_positions);
            hipFree(d_velocities);
            hipFree(d_forces);
            return -1;
        }

        // Synchronize to ensure all kernels are complete
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::cerr << "Error synchronizing device: " << hipGetErrorString(err) << std::endl;
            hipFree(d_positions);
            hipFree(d_velocities);
            hipFree(d_forces);
            return -1;
        }
    }

    // Free device memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_forces);

    std::cout << "Cloth simulation completed successfully." << std::endl;
    return 0;
}