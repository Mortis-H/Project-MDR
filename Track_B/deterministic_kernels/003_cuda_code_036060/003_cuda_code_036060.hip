#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 2048
#define BLOCK_SIZE 256

__global__ void initParticles(hiprandState *states, float4 *positions, float4 *velocities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        hiprandState localState = states[idx];
        positions[idx].x = hiprand_uniform(&localState) * 10.0f;
        positions[idx].y = hiprand_uniform(&localState) * 10.0f;
        positions[idx].z = hiprand_uniform(&localState) * 10.0f;
        positions[idx].w = 0.0f; // Mass or charge

        velocities[idx].x = hiprand_uniform(&localState) * 1.0f - 0.5f;
        velocities[idx].y = hiprand_uniform(&localState) * 1.0f - 0.5f;
        velocities[idx].z = hiprand_uniform(&localState) * 1.0f - 0.5f;
        velocities[idx].w = 0.0f; // Padding

        states[idx] = localState;
    }
}

__global__ void computeForces(float4 *positions, float4 *forces, float dt) {
    extern __shared__ float4 sharedPositions[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    if (idx < N) {
        sharedPositions[tid] = positions[idx];
    }
    __syncthreads();

    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Compute forces using shared memory
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        float4 other = sharedPositions[i];
        float dx = positions[idx].x - other.x;
        float dy = positions[idx].y - other.y;
        float dz = positions[idx].z - other.z;
        float r2 = dx * dx + dy * dy + dz * dz + 1e-6f; // Avoid division by zero
        float r = sqrtf(r2);
        float inv_r3 = 1.0f / (r * r2);

        force.x += dx * inv_r3;
        force.y += dy * inv_r3;
        force.z += dz * inv_r3;
    }
    __syncthreads();

    // Write back the computed force
    if (idx < N) {
        forces[idx] = force;
    }
}

__global__ void updateParticles(float4 *positions, float4 *velocities, float4 *forces, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Update velocity and position
        velocities[idx].x += forces[idx].x * dt;
        velocities[idx].y += forces[idx].y * dt;
        velocities[idx].z += forces[idx].z * dt;

        positions[idx].x += velocities[idx].x * dt;
        positions[idx].y += velocities[idx].y * dt;
        positions[idx].z += velocities[idx].z * dt;
    }
}

int main() {
    float4 *d_positions, *d_velocities, *d_forces;
    hiprandState *d_states;
    float dt = 0.01f;

    // Allocate memory
    hipMalloc((void **)&d_positions, N * sizeof(float4));
    hipMalloc((void **)&d_velocities, N * sizeof(float4));
    hipMalloc((void **)&d_forces, N * sizeof(float4));
    hipMalloc((void **)&d_states, N * sizeof(hiprandState));

    // Initialize random states
    hiprandState *d_statesInit;
    hipMalloc((void **)&d_statesInit, N * sizeof(hiprandState));
    initParticles<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_statesInit, d_positions, d_velocities);
    hipMemcpy(d_states, d_statesInit, N * sizeof(hiprandState), hipMemcpyDeviceToDevice);
    hipFree(d_statesInit);

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Simulation loop
    for (int step = 0; step < 100; ++step) {
        computeForces<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float4)>>>(d_positions, d_forces, dt);
        updateParticles<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_positions, d_velocities, d_forces, dt);

        // Check for errors
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Free memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_forces);
    hipFree(d_states);

    return 0;
}