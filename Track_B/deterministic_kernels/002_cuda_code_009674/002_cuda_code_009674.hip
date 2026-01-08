#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARTICLES 128
#define DT 0.001f
#define NUM_STEPS 1000

__global__ void initializeParticles(float4 *pos, float4 *vel, float4 *acc, hiprandState *states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        hiprandState localState = states[idx];
        hiprand_init(seed, idx, 0, &localState);
        pos[idx] = make_float4(hiprand_uniform(&localState) * 10.0f, hiprand_uniform(&localState) * 10.0f, hiprand_uniform(&localState) * 10.0f, 0.0f);
        vel[idx] = make_float4(hiprand_uniform(&localState) * 0.1f, hiprand_uniform(&localState) * 0.1f, hiprand_uniform(&localState) * 0.1f, 0.0f);
        acc[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        states[idx] = localState;
    }
}

__device__ float4 computeForce(float4 pos, float4 *positions, int numParticles) {
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j = 0; j < numParticles; j++) {
        float4 r = make_float4(positions[j].x - pos.x, positions[j].y - pos.y, positions[j].z - pos.z, 0.0f);
        float distSquared = r.x * r.x + r.y * r.y + r.z * r.z + 1e-5f;
        float dist = sqrtf(distSquared);
        float forceMagnitude = 1.0f / (distSquared * dist);
        force.x += forceMagnitude * r.x / dist;
        force.y += forceMagnitude * r.y / dist;
        force.z += forceMagnitude * r.z / dist;
    }
    return force;
}

__global__ void leapfrogStep(float4 *pos, float4 *vel, float4 *acc, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        // First half kick
        vel[idx].x += 0.5f * DT * acc[idx].x;
        vel[idx].y += 0.5f * DT * acc[idx].y;
        vel[idx].z += 0.5f * DT * acc[idx].z;

        // Drift
        pos[idx].x += DT * vel[idx].x;
        pos[idx].y += DT * vel[idx].y;
        pos[idx].z += DT * vel[idx].z;

        // Compute new acceleration
        float4 force = computeForce(pos[idx], pos, numParticles);
        acc[idx].x = force.x;
        acc[idx].y = force.y;
        acc[idx].z = force.z;

        // Second half kick
        vel[idx].x += 0.5f * DT * acc[idx].x;
        vel[idx].y += 0.5f * DT * acc[idx].y;
        vel[idx].z += 0.5f * DT * acc[idx].z;
    }
}

int main() {
    float4 *d_pos, *d_vel, *d_acc;
    hiprandState *d_states;

    // Allocate memory on the device
    hipMalloc(&d_pos, NUM_PARTICLES * sizeof(float4));
    hipMalloc(&d_vel, NUM_PARTICLES * sizeof(float4));
    hipMalloc(&d_acc, NUM_PARTICLES * sizeof(float4));
    hipMalloc(&d_states, NUM_PARTICLES * sizeof(hiprandState));

    // Initialize particles
    initializeParticles<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_pos, d_vel, d_acc, d_states, 1234);
    hipDeviceSynchronize();

    // Leapfrog integration
    for (int step = 0; step < NUM_STEPS; step++) {
        leapfrogStep<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_pos, d_vel, d_acc, NUM_PARTICLES);
        hipDeviceSynchronize();
    }

    // Free device memory
    hipFree(d_pos);
    hipFree(d_vel);
    hipFree(d_acc);
    hipFree(d_states);

    return 0;
}