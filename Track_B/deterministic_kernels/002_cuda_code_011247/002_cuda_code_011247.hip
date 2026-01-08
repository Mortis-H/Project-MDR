#include <iostream>
#include <hip/hip_runtime.h>
#include <vector>

// Constants
#define NUM_PARTICLES 8192
#define BLOCK_SIZE 256
#define DT 0.001f
#define EPSILON 0.01f
#define SIGMA 1.0f

// Kernel to compute forces using Lennard-Jones potential
__global__ void computeForces(float4 *pos, float4 *force, int numParticles) {
    extern __shared__ float4 s_pos[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load position data into shared memory to avoid bank conflicts
    s_pos[tid] = pos[idx];
    __syncthreads();

    float4 f = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = 0; j < blockDim.x; ++j) {
        if (idx != j) {
            float4 r = {s_pos[j].x - s_pos[tid].x, s_pos[j].y - s_pos[tid].y, s_pos[j].z - s_pos[tid].z, 0.0f};
            float rsq = r.x * r.x + r.y * r.y + r.z * r.z;
            float r6 = rsq * rsq * rsq;
            float r12 = r6 * r6;
            float lj = 48.0f * (1.0f / r12 - 1.0f / r6) / rsq;

            f.x += lj * r.x;
            f.y += lj * r.y;
            f.z += lj * r.z;
        }
    }
    __syncthreads();

    // Accumulate forces from shared memory
    atomicAdd(&force[idx].x, f.x);
    atomicAdd(&force[idx].y, f.y);
    atomicAdd(&force[idx].z, f.z);
}

// Kernel to update positions and velocities using Verlet integration
__global__ void updateParticles(float4 *pos, float4 *vel, float4 *force, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        float4 a = {force[idx].x, force[idx].y, force[idx].z, 0.0f};
        float4 vhalf = {vel[idx].x + 0.5f * DT * a.x, vel[idx].y + 0.5f * DT * a.y, vel[idx].z + 0.5f * DT * a.z, 0.0f};
        float4 pnew = {pos[idx].x + DT * vhalf.x, pos[idx].y + DT * vhalf.y, pos[idx].z + DT * vhalf.z, 0.0f};
        float4 anew = {force[idx].x, force[idx].y, force[idx].z, 0.0f};
        float4 vnew = {vhalf.x + 0.5f * DT * anew.x, vhalf.y + 0.5f * DT * anew.y, vhalf.z + 0.5f * DT * anew.z, 0.0f};

        pos[idx] = pnew;
        vel[idx] = vnew;
        force[idx] = {0.0f, 0.0f, 0.0f, 0.0f}; // Reset force for next step
    }
}

int main() {
    // Initialize host data
    std::vector<float4> h_pos(NUM_PARTICLES);
    std::vector<float4> h_vel(NUM_PARTICLES);
    std::vector<float4> h_force(NUM_PARTICLES);

    // Initialize positions and velocities (simple initialization)
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_pos[i] = {static_cast<float>(i) * EPSILON, 0.0f, 0.0f, 0.0f};
        h_vel[i] = {0.0f, 0.0f, 0.0f, 0.0f};
        h_force[i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    // Allocate device memory
    float4 *d_pos, *d_vel, *d_force;
    hipMalloc(&d_pos, NUM_PARTICLES * sizeof(float4));
    hipMalloc(&d_vel, NUM_PARTICLES * sizeof(float4));
    hipMalloc(&d_force, NUM_PARTICLES * sizeof(float4));

    // Copy data to device
    hipMemcpy(d_pos, h_pos.data(), NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_vel, h_vel.data(), NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_force, h_force.data(), NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);

    // Set up execution configuration
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((NUM_PARTICLES + threads.x - 1) / threads.x);

    // Perform a few steps of the simulation
    for (int step = 0; step < 100; ++step) {
        computeForces<<<blocks, threads, threads.x * sizeof(float4)>>>(d_pos, d_force, NUM_PARTICLES);
        hipDeviceSynchronize();
        updateParticles<<<blocks, threads>>>(d_pos, d_vel, d_force, NUM_PARTICLES);
        hipDeviceSynchronize();
    }

    // Copy results back to host
    hipMemcpy(h_pos.data(), d_pos, NUM_PARTICLES * sizeof(float4), hipMemcpyDeviceToHost);
    hipMemcpy(h_vel.data(), d_vel, NUM_PARTICLES * sizeof(float4), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_pos);
    hipFree(d_vel);
    hipFree(d_force);

    // Check for any errors launching the kernel
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    return 0;
}