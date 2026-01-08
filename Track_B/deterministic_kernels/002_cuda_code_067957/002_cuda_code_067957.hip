#include <iostream>
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>

// Constants
const int NUM_PARTICLES = 32;
const float DT = 0.005f; // Timestep
const float EPSILON = 1.0f; // Lennard-Jones epsilon
const float SIGMA = 1.0f; // Lennard-Jones sigma
const float RCUT = 2.5f; // Cutoff radius

// Kernel to compute forces using Lennard-Jones potential
__global__ void computeForces(float4* pos, float4* forces, float4* vel, float4* acc) {
    extern __shared__ float4 sharedPos[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    // Load positions into shared memory
    sharedPos[threadIdx.x] = pos[idx];
    __syncthreads();

    float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Compute forces using Lennard-Jones potential
    for (int j = 0; j < blockDim.x; ++j) {
        if (j != threadIdx.x) {
            float3 dr = make_float3(sharedPos[j].x - sharedPos[threadIdx.x].x,
                                   sharedPos[j].y - sharedPos[threadIdx.x].y,
                                   sharedPos[j].z - sharedPos[threadIdx.x].z);
            float r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

            if (r2 < RCUT * RCUT) {
                float r6 = r2 * r2 * r2;
                float fmag = 24.0f * EPSILON * ((2.0f / r6) * (2.0f / r6) - (1.0f / r6)) / r2;
                f.x += fmag * dr.x;
                f.y += fmag * dr.y;
                f.z += fmag * dr.z;
            }
        }
    }
    __syncthreads();

    // Store forces
    forces[idx] = f;
}

// Kernel to update positions and velocities using Velocity Verlet
__global__ void updatePositionsAndVelocities(float4* pos, float4* forces, float4* vel, float4* acc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    // Update velocity and position
    float4 a = forces[idx];
    vel[idx].x += DT * 0.5f * a.x;
    vel[idx].y += DT * 0.5f * a.y;
    vel[idx].z += DT * 0.5f * a.z;

    pos[idx].x += DT * vel[idx].x;
    pos[idx].y += DT * vel[idx].y;
    pos[idx].z += DT * vel[idx].z;

    // Update acceleration
    acc[idx] = a;
}

// Main function to demonstrate the usage of the kernels
int main() {
    // Allocate memory for positions, velocities, forces, and accelerations
    float4* d_pos, *d_vel, *d_forces, *d_acc;
    hipMalloc(&d_pos, NUM_PARTICLES * sizeof(float4));
    hipMalloc(&d_vel, NUM_PARTICLES * sizeof(float4));
    hipMalloc(&d_forces, NUM_PARTICLES * sizeof(float4));
    hipMalloc(&d_acc, NUM_PARTICLES * sizeof(float4));

    // Initialize positions and velocities
    std::vector<float4> h_pos(NUM_PARTICLES);
    std::vector<float4> h_vel(NUM_PARTICLES);
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_pos[i] = make_float4(i, i, i, 0.0f);
        h_vel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    hipMemcpy(d_pos, h_pos.data(), NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_vel, h_vel.data(), NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);

    // Set up block and grid dimensions
    int blockSize = 32;
    int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;

    // Perform a few steps of molecular dynamics
    for (int step = 0; step < 10; ++step) {
        computeForces<<<numBlocks, blockSize, blockSize * sizeof(float4)>>>(d_pos, d_forces, d_vel, d_acc);
        hipDeviceSynchronize();
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error in computeForces: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        updatePositionsAndVelocities<<<numBlocks, blockSize>>>(d_pos, d_forces, d_vel, d_acc);
        hipDeviceSynchronize();
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error in updatePositionsAndVelocities: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy final positions back to host
    std::vector<float4> final_pos(NUM_PARTICLES);
    hipMemcpy(final_pos.data(), d_pos, NUM_PARTICLES * sizeof(float4), hipMemcpyDeviceToHost);

    // Clean up
    hipFree(d_pos);
    hipFree(d_vel);
    hipFree(d_forces);
    hipFree(d_acc);

    return 0;
}