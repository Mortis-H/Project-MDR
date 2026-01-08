#include <iostream>
#include <hip/hip_runtime.h>

// Constants
#define N 8192
#define BLOCK_SIZE 256
#define DT 0.001f
#define EPSILON 1.0f
#define SIGMA 1.0f

// Kernel to calculate forces using Lennard-Jones potential
__global__ void calculateForces(float4* pos, float4* force, float4* vel) {
    extern __shared__ float4 sharedPos[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    // Load position data into shared memory
    if (idx < N) {
        sharedPos[tid] = pos[idx];
    }
    __syncthreads();

    // Initialize force to zero
    float4 f = {0.0f, 0.0f, 0.0f, 0.0f};

    // Compute forces using Lennard-Jones potential
    for (int i = 0; i < blockDim.x; ++i) {
        if (idx != bid * blockDim.x + i) {
            float4 r;
            r.x = sharedPos[i].x - sharedPos[tid].x;
            r.y = sharedPos[i].y - sharedPos[tid].y;
            r.z = sharedPos[i].z - sharedPos[tid].z;

            float r2 = r.x * r.x + r.y * r.y + r.z * r.z;
            float r6 = r2 * r2 * r2;
            float r12 = r6 * r6;

            float ljForce = 24.0f * EPSILON * ((2.0f * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA / r12) - (SIGMA * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA / r6)) / r2;

            f.x += ljForce * r.x;
            f.y += ljForce * r.y;
            f.z += ljForce * r.z;
        }
    }
    __syncthreads();

    // Write force back to global memory
    if (idx < N) {
        force[idx] = f;
    }
}

// Kernel to update positions and velocities using velocity Verlet algorithm
__global__ void updateParticles(float4* pos, float4* force, float4* vel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Update velocity at half time step
        vel[idx].x += 0.5f * DT * force[idx].x;
        vel[idx].y += 0.5f * DT * force[idx].y;
        vel[idx].z += 0.5f * DT * force[idx].z;

        // Update position
        pos[idx].x += DT * vel[idx].x;
        pos[idx].y += DT * vel[idx].y;
        pos[idx].z += DT * vel[idx].z;
    }
}

int main() {
    // Allocate host memory
    float4* h_pos = new float4[N];
    float4* h_force = new float4[N];
    float4* h_vel = new float4[N];

    // Initialize particle positions and velocities
    for (int i = 0; i < N; ++i) {
        h_pos[i] = {static_cast<float>(i) * 0.1f, 0.0f, 0.0f, 0.0f};
        h_vel[i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    // Allocate device memory
    float4* d_pos, *d_force, *d_vel;
    hipMalloc((void**)&d_pos, N * sizeof(float4));
    hipMalloc((void**)&d_force, N * sizeof(float4));
    hipMalloc((void**)&d_vel, N * sizeof(float4));

    // Copy data from host to device
    hipMemcpy(d_pos, h_pos, N * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_vel, h_vel, N * sizeof(float4), hipMemcpyHostToDevice);

    // Launch kernels
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int step = 0; step < 10; ++step) {
        calculateForces<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float4)>>>(d_pos, d_force, d_vel);
        hipDeviceSynchronize();
        updateParticles<<<numBlocks, BLOCK_SIZE>>>(d_pos, d_force, d_vel);
        hipDeviceSynchronize();
    }

    // Copy results back to host
    hipMemcpy(h_pos, d_pos, N * sizeof(float4), hipMemcpyDeviceToHost);
    hipMemcpy(h_vel, d_vel, N * sizeof(float4), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_pos);
    hipFree(d_force);
    hipFree(d_vel);

    // Free host memory
    delete[] h_pos;
    delete[] h_force;
    delete[] h_vel;

    // Check for any errors launching the kernel
    hipError_t cudaStatus = hipGetLastError();
    if (cudaStatus != hipSuccess) {
        fprintf(stderr, "calculateForces or updateParticles launch failed: %s\n", hipGetErrorString(cudaStatus));
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight Systems or Nsight Compute to show complete traces.
    cudaStatus = hipDeviceReset();
    if (cudaStatus != hipSuccess) {
        fprintf(stderr, "hipDeviceReset failed!");
        return 1;
    }

    return 0;
}