#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 4096
#define DT 0.001f
#define NUM_STEPS 100

// Kernel to initialize positions and velocities
__global__ void initialize(hiprandState *state, float4 *positions, float4 *velocities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        hiprandState localState = state[idx];
        float x = hiprand_uniform(&localState) * 10.0f;
        float y = hiprand_uniform(&localState) * 10.0f;
        float z = hiprand_uniform(&localState) * 10.0f;
        float vx = hiprand_uniform(&localState) * 0.1f;
        float vy = hiprand_uniform(&localState) * 0.1f;
        float vz = hiprand_uniform(&localState) * 0.1f;
        positions[idx] = make_float4(x, y, z, 0.0f);
        velocities[idx] = make_float4(vx, vy, vz, 0.0f);
        state[idx] = localState;
    }
}

// Simple Lennard-Jones potential for force calculation
__device__ float4 computeForce(const float4 &pos, const float4 *positions) {
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j = 0; j < N; j++) {
        float dx = pos.x - positions[j].x;
        float dy = pos.y - positions[j].y;
        float dz = pos.z - positions[j].z;
        float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > 0.0f && r2 < 1.0f) {
            float r6 = r2 * r2 * r2;
            float r12 = r6 * r6;
            float f = 24.0f * (2.0f / r12 - 1.0f / r6) / r2;
            force.x += f * dx;
            force.y += f * dy;
            force.z += f * dz;
        }
    }
    return force;
}

// Leapfrog integration kernel
__global__ void leapfrog(float4 *positions, float4 *velocities, float4 *forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Update positions
        positions[idx].x += velocities[idx].x * DT + 0.5f * forces[idx].x * DT * DT;
        positions[idx].y += velocities[idx].y * DT + 0.5f * forces[idx].y * DT * DT;
        positions[idx].z += velocities[idx].z * DT + 0.5f * forces[idx].z * DT * DT;

        // Compute new forces
        float4 newForce = computeForce(positions[idx], positions);

        // Update velocities
        velocities[idx].x += 0.5f * (forces[idx].x + newForce.x) * DT;
        velocities[idx].y += 0.5f * (forces[idx].y + newForce.y) * DT;
        velocities[idx].z += 0.5f * (forces[idx].z + newForce.z) * DT;

        // Store new forces
        forces[idx] = newForce;
    }
}

int main() {
    float4 *d_positions, *d_velocities, *d_forces;
    hiprandState *d_states;

    // Allocate memory on the device
    hipMalloc((void**)&d_positions, N * sizeof(float4));
    hipMalloc((void**)&d_velocities, N * sizeof(float4));
    hipMalloc((void**)&d_forces, N * sizeof(float4));
    hipMalloc((void**)&d_states, N * sizeof(hiprandState));

    // Initialize random states
    unsigned int seed = 1234;
    hiprandState *d_state;
    hipMalloc((void**)&d_state, N * sizeof(hiprandState));
    hipMemset(d_state, 0, N * sizeof(hiprandState));
    initialize<<<(N + 255) / 256, 256>>>(d_state, d_positions, d_velocities);
    hipDeviceSynchronize();

    // Initialize forces
    leapfrog<<<(N + 255) / 256, 256>>>(d_positions, d_velocities, d_forces);
    hipDeviceSynchronize();

    // Perform leapfrog integration for NUM_STEPS
    for (int step = 0; step < NUM_STEPS; step++) {
        leapfrog<<<(N + 255) / 256, 256>>>(d_positions, d_velocities, d_forces);
        hipDeviceSynchronize();
    }

    // Free device memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_forces);
    hipFree(d_states);

    // Check for any errors launching the kernel
    hipError_t cudaStatus = hipGetLastError();
    if (cudaStatus != hipSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", hipGetErrorString(cudaStatus));
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight Systems or Nsight Compute to show complete traces.
    cudaStatus = hipDeviceReset();
    if (cudaStatus != hipSuccess) {
        fprintf(stderr, "hipDeviceReset failed!\n");
        return 1;
    }

    return 0;
}