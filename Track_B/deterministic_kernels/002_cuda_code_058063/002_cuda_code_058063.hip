// nbody_simulation.cu
// A simple N-body simulation using brute force method on a GPU with persistent threads.
// Target architecture: Hopper

#include <iostream>
#include <hip/hip_runtime.h>
#include <cmath>

// Constants
#define N 512
#define G 0.0000000000667430f // Gravitational constant
#define DT 0.01f              // Time step
#define EPS 0.0001f           // Softening factor

// Kernel to compute forces
__global__ void computeForces(float4* pos, float4* acc, float* mass) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int j = 0; j < N; j++) {
            if (i != j) {
                float4 r = make_float4(pos[j].x - pos[i].x, pos[j].y - pos[i].y, pos[j].z - pos[i].z, 0.0f);
                float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS * EPS;
                float invDist = rsqrtf(distSqr);
                float invDistCube = invDist * invDist * invDist;
                float s = G * mass[j] * invDistCube;
                force.x += r.x * s;
                force.y += r.y * s;
                force.z += r.z * s;
            }
        }
        acc[i] = force;
    }
}

// Kernel to update positions
__global__ void updatePositions(float4* pos, float4* vel, float4* acc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        vel[i].x += acc[i].x * DT;
        vel[i].y += acc[i].y * DT;
        vel[i].z += acc[i].z * DT;
        pos[i].x += vel[i].x * DT;
        pos[i].y += vel[i].y * DT;
        pos[i].z += vel[i].z * DT;
    }
}

int main() {
    // Allocate memory for positions, velocities, accelerations, and masses
    float4* d_pos;
    float4* d_vel;
    float4* d_acc;
    float* d_mass;

    hipMalloc((void**)&d_pos, N * sizeof(float4));
    hipMalloc((void**)&d_vel, N * sizeof(float4));
    hipMalloc((void**)&d_acc, N * sizeof(float4));
    hipMalloc((void**)&d_mass, N * sizeof(float));

    // Initialize positions, velocities, and masses
    float4* h_pos = new float4[N];
    float4* h_vel = new float4[N];
    float* h_mass = new float[N];

    for (int i = 0; i < N; i++) {
        h_pos[i] = make_float4(static_cast<float>(i) / N, static_cast<float>(i) / N, static_cast<float>(i) / N, 0.0f);
        h_vel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        h_mass[i] = 1.0f;
    }

    hipMemcpy(d_pos, h_pos, N * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_vel, h_vel, N * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_mass, h_mass, N * sizeof(float), hipMemcpyHostToDevice);

    // Set up execution configuration
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Simulate for a number of steps
    int steps = 10;
    for (int step = 0; step < steps; step++) {
        // Compute forces
        computeForces<<<numBlocks, blockSize>>>(d_pos, d_acc, d_mass);
        hipDeviceSynchronize();
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in computeForces kernel: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Update positions
        updatePositions<<<numBlocks, blockSize>>>(d_pos, d_vel, d_acc);
        hipDeviceSynchronize();
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in updatePositions kernel: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy results back to host
    hipMemcpy(h_pos, d_pos, N * sizeof(float4), hipMemcpyDeviceToHost);

    // Output the final positions (first 10 particles)
    for (int i = 0; i < 10; i++) {
        std::cout << "Particle " << i << ": (" << h_pos[i].x << ", " << h_pos[i].y << ", " << h_pos[i].z << ")" << std::endl;
    }

    // Free memory
    delete[] h_pos;
    delete[] h_vel;
    delete[] h_mass;
    hipFree(d_pos);
    hipFree(d_vel);
    hipFree(d_acc);
    hipFree(d_mass);

    return 0;
}