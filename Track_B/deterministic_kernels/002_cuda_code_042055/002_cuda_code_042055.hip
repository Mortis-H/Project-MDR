#include <iostream>
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>

// Constants
#define NUM_PARTICLES 512
#define SOFTENING 1e-9f
#define G 6.67384e-11f

// CUDA kernel to calculate forces
__global__ void nbodyKernel(float4 *pos, float4 *vel, float4 *acc, float dt, int numParticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 p = pos[i];
        for (int j = 0; j < numParticles; j++) {
            if (i != j) {
                float4 q = pos[j];
                float dx = q.x - p.x;
                float dy = q.y - p.y;
                float dz = q.z - p.z;
                float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                force.x += dx * invDist3;
                force.y += dy * invDist3;
                force.z += dz * invDist3;
            }
        }
        atomicAdd(&acc[i].x, force.x * G);
        atomicAdd(&acc[i].y, force.y * G);
        atomicAdd(&acc[i].z, force.z * G);
    }
}

// Function to update positions and velocities
__global__ void updateKernel(float4 *pos, float4 *vel, float4 *acc, float dt, int numParticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        vel[i].x += acc[i].x * dt;
        vel[i].y += acc[i].y * dt;
        vel[i].z += acc[i].z * dt;
        pos[i].x += vel[i].x * dt;
        pos[i].y += vel[i].y * dt;
        pos[i].z += vel[i].z * dt;
        acc[i].x = 0.0f;
        acc[i].y = 0.0f;
        acc[i].z = 0.0f;
    }
}

int main() {
    // Initialize host data
    std::vector<float4> h_pos(NUM_PARTICLES);
    std::vector<float4> h_vel(NUM_PARTICLES);
    std::vector<float4> h_acc(NUM_PARTICLES);

    // Initialize particles randomly
    for (int i = 0; i < NUM_PARTICLES; i++) {
        h_pos[i] = make_float4(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, 0.0f);
        h_vel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        h_acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Allocate device memory
    float4 *d_pos, *d_vel, *d_acc;
    hipMalloc(&d_pos, NUM_PARTICLES * sizeof(float4));
    hipMalloc(&d_vel, NUM_PARTICLES * sizeof(float4));
    hipMalloc(&d_acc, NUM_PARTICLES * sizeof(float4));

    // Copy data to device
    hipMemcpy(d_pos, h_pos.data(), NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_vel, h_vel.data(), NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_acc, h_acc.data(), NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);

    // Define grid and block sizes
    int blockSize = 256;
    int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;

    // Simulation parameters
    float dt = 0.01f;
    int numSteps = 100;

    // Main simulation loop
    for (int step = 0; step < numSteps; step++) {
        nbodyKernel<<<numBlocks, blockSize>>>(d_pos, d_vel, d_acc, dt, NUM_PARTICLES);
        hipDeviceSynchronize();
        updateKernel<<<numBlocks, blockSize>>>(d_pos, d_vel, d_acc, dt, NUM_PARTICLES);
        hipDeviceSynchronize();
    }

    // Copy results back to host
    hipMemcpy(h_pos.data(), d_pos, NUM_PARTICLES * sizeof(float4), hipMemcpyDeviceToHost);
    hipMemcpy(h_vel.data(), d_vel, NUM_PARTICLES * sizeof(float4), hipMemcpyDeviceToHost);
    hipMemcpy(h_acc.data(), d_acc, NUM_PARTICLES * sizeof(float4), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_pos);
    hipFree(d_vel);
    hipFree(d_acc);

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "N-body simulation completed successfully." << std::endl;
    return 0;
}