#include <iostream>
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>

// Constants
#define NUM_PARTICLES 256
#define DIM 3
#define DT 0.001f
#define EPSILON 1e-5f
#define GRAVITY 9.81f

// CUDA kernel for Verlet integration
__global__ void verletIntegration(float* pos, float* vel, float* acc, float* newAcc, float* forces, float dt, int numParticles) {
    extern __shared__ float sharedMem[];
    float* sharedPos = sharedMem;
    float* sharedAcc = &sharedMem[numParticles * DIM];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // Load position and acceleration into shared memory
    for (int d = 0; d < DIM; ++d) {
        sharedPos[idx * DIM + d] = pos[idx * DIM + d];
        sharedAcc[idx * DIM + d] = acc[idx * DIM + d];
    }
    __syncthreads();

    // Verlet integration step
    for (int d = 0; d < DIM; ++d) {
        pos[idx * DIM + d] = sharedPos[idx * DIM + d] + vel[idx * DIM + d] * dt + 0.5f * sharedAcc[idx * DIM + d] * dt * dt;
        vel[idx * DIM + d] = vel[idx * DIM + d] + 0.5f * (sharedAcc[idx * DIM + d] + newAcc[idx * DIM + d]) * dt;
    }

    // Store new acceleration
    for (int d = 0; d < DIM; ++d) {
        acc[idx * DIM + d] = newAcc[idx * DIM + d];
    }
}

// Kernel to compute forces and new accelerations
__global__ void computeForcesAndAcc(float* pos, float* newAcc, float* forces, float* mass, int numParticles) {
    extern __shared__ float sharedMem[];
    float* sharedPos = sharedMem;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // Load position into shared memory
    for (int d = 0; d < DIM; ++d) {
        sharedPos[idx * DIM + d] = pos[idx * DIM + d];
    }
    __syncthreads();

    // Zero out forces
    for (int d = 0; d < DIM; ++d) {
        forces[idx * DIM + d] = 0.0f;
    }

    // Compute forces using a simple gravity model for demonstration
    for (int j = 0; j < numParticles; ++j) {
        if (j != idx) {
            float dx[DIM];
            float r2 = 0.0f;
            for (int d = 0; d < DIM; ++d) {
                dx[d] = sharedPos[j * DIM + d] - sharedPos[idx * DIM + d];
                r2 += dx[d] * dx[d];
            }
            r2 += EPSILON; // Prevent division by zero
            float r = sqrtf(r2);
            float f = GRAVITY / (r2 * r); // Simplified force calculation
            for (int d = 0; d < DIM; ++d) {
                forces[idx * DIM + d] += f * dx[d];
            }
        }
    }

    // Compute new acceleration
    for (int d = 0; d < DIM; ++d) {
        newAcc[idx * DIM + d] = forces[idx * DIM + d] / mass[idx];
    }
}

int main() {
    // Initialize host data
    std::vector<float> h_pos(NUM_PARTICLES * DIM);
    std::vector<float> h_vel(NUM_PARTICLES * DIM);
    std::vector<float> h_acc(NUM_PARTICLES * DIM);
    std::vector<float> h_newAcc(NUM_PARTICLES * DIM);
    std::vector<float> h_forces(NUM_PARTICLES * DIM);
    std::vector<float> h_mass(NUM_PARTICLES, 1.0f); // Assuming all particles have the same mass

    // Initialize positions and velocities
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        for (int d = 0; d < DIM; ++d) {
            h_pos[i * DIM + d] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
            h_vel[i * DIM + d] = static_cast<float>(rand()) / RAND_MAX * 0.1f;
        }
    }

    // Allocate device memory
    float* d_pos, *d_vel, *d_acc, *d_newAcc, *d_forces, *d_mass;
    hipMalloc(&d_pos, NUM_PARTICLES * DIM * sizeof(float));
    hipMalloc(&d_vel, NUM_PARTICLES * DIM * sizeof(float));
    hipMalloc(&d_acc, NUM_PARTICLES * DIM * sizeof(float));
    hipMalloc(&d_newAcc, NUM_PARTICLES * DIM * sizeof(float));
    hipMalloc(&d_forces, NUM_PARTICLES * DIM * sizeof(float));
    hipMalloc(&d_mass, NUM_PARTICLES * sizeof(float));

    // Copy data to device
    hipMemcpy(d_pos, h_pos.data(), NUM_PARTICLES * DIM * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_vel, h_vel.data(), NUM_PARTICLES * DIM * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_acc, h_acc.data(), NUM_PARTICLES * DIM * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_mass, h_mass.data(), NUM_PARTICLES * sizeof(float), hipMemcpyHostToDevice);

    // Set grid and block dimensions
    int blockSize = 256;
    int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;

    // Perform simulation steps
    for (int step = 0; step < 100; ++step) {
        // Compute forces and new accelerations
        computeForcesAndAcc<<<numBlocks, blockSize, NUM_PARTICLES * DIM * sizeof(float) * 2>>>(d_pos, d_newAcc, d_forces, d_mass, NUM_PARTICLES);
        hipDeviceSynchronize();
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in computeForcesAndAcc: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Perform Verlet integration
        verletIntegration<<<numBlocks, blockSize, NUM_PARTICLES * DIM * sizeof(float) * 2>>>(d_pos, d_vel, d_acc, d_newAcc, d_forces, DT, NUM_PARTICLES);
        hipDeviceSynchronize();
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in verletIntegration: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy results back to host
    hipMemcpy(h_pos.data(), d_pos, NUM_PARTICLES * DIM * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_vel.data(), d_vel, NUM_PARTICLES * DIM * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_pos);
    hipFree(d_vel);
    hipFree(d_acc);
    hipFree(d_newAcc);
    hipFree(d_forces);
    hipFree(d_mass);

    return 0;
}