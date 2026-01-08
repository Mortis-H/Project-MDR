#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 256
#define DT 0.001f
#define RCUT 3.0f
#define EPS 1.0f
#define SIGMA 1.0f

// Kernel to initialize random positions and velocities
__global__ void initialize(hiprandState *states, float4 *positions, float4 *velocities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        hiprandState localState = states[idx];
        float x = hiprand_uniform(&localState) * 10.0f;
        float y = hiprand_uniform(&localState) * 10.0f;
        float z = hiprand_uniform(&localState) * 10.0f;
        positions[idx] = make_float4(x, y, z, 0.0f);
        float vx = (hiprand_uniform(&localState) - 0.5f) * 2.0f;
        float vy = (hiprand_uniform(&localState) - 0.5f) * 2.0f;
        float vz = (hiprand_uniform(&localState) - 0.5f) * 2.0f;
        velocities[idx] = make_float4(vx, vy, vz, 0.0f);
        states[idx] = localState;
    }
}

// Kernel to calculate forces using Verlet list
__global__ void computeForces(float4 *positions, float4 *forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float4 pos = positions[idx];
        float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int j = 0; j < N; ++j) {
            if (j != idx) {
                float4 posJ = positions[j];
                float dx = pos.x - posJ.x;
                float dy = pos.y - posJ.y;
                float dz = pos.z - posJ.z;
                float r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < RCUT * RCUT) {
                    float r6 = r2 * r2 * r2;
                    float r12 = r6 * r6;
                    float forceFactor = 24.0f * EPS * (2.0f * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA / r12 - 1.0f / r6) / r2;
                    force.x += forceFactor * dx;
                    force.y += forceFactor * dy;
                    force.z += forceFactor * dz;
                }
            }
        }
        forces[idx] = force;
    }
}

// Kernel to update positions and velocities using Verlet integration
__global__ void updatePositionsAndVelocities(float4 *positions, float4 *velocities, float4 *forces, float4 *newForces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float4 pos = positions[idx];
        float4 vel = velocities[idx];
        float4 force = forces[idx];
        float4 newForce = newForces[idx];
        pos.x += vel.x * DT + 0.5f * force.x * DT * DT;
        pos.y += vel.y * DT + 0.5f * force.y * DT * DT;
        pos.z += vel.z * DT + 0.5f * force.z * DT * DT;
        vel.x += 0.5f * (force.x + newForce.x) * DT;
        vel.y += 0.5f * (force.y + newForce.y) * DT;
        vel.z += 0.5f * (force.z + newForce.z) * DT;
        positions[idx] = pos;
        velocities[idx] = vel;
    }
}

int main() {
    float4 *d_positions, *d_velocities, *d_forces, *d_newForces;
    hiprandState *d_states;

    // Allocate memory
    hipMalloc(&d_positions, N * sizeof(float4));
    hipMalloc(&d_velocities, N * sizeof(float4));
    hipMalloc(&d_forces, N * sizeof(float4));
    hipMalloc(&d_newForces, N * sizeof(float4));
    hipMalloc(&d_states, N * sizeof(hiprandState));

    // Initialize random states
    unsigned int seed = 1234;
    hiprandState *devStates;
    hipMalloc(&devStates, N * sizeof(hiprandState));
    initialize<<<(N + 255) / 256, 256>>>(devStates, d_positions, d_velocities);
    hipDeviceSynchronize();

    // Simulation loop
    int numSteps = 100;
    for (int step = 0; step < numSteps; ++step) {
        computeForces<<<(N + 255) / 256, 256>>>(d_positions, d_forces);
        hipDeviceSynchronize();

        updatePositionsAndVelocities<<<(N + 255) / 256, 256>>>(d_positions, d_velocities, d_forces, d_newForces);
        hipDeviceSynchronize();

        // Swap forces for next step
        float4 *temp = d_forces;
        d_forces = d_newForces;
        d_newForces = temp;
    }

    // Free memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_forces);
    hipFree(d_newForces);
    hipFree(d_states);

    return 0;
}