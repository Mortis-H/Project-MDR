#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define BLOCK_SIZE 256
#define NUM_PARTICLES 1024

__global__ void initializeParticles(hiprandState *states, float4 *positions, float4 *velocities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        hiprandState localState = states[idx];
        positions[idx].x = hiprand_uniform(&localState) * 10.0f;
        positions[idx].y = hiprand_uniform(&localState) * 10.0f;
        positions[idx].z = hiprand_uniform(&localState) * 10.0f;
        positions[idx].w = 0.0f; // Mass
        velocities[idx].x = hiprand_uniform(&localState) * 2.0f - 1.0f;
        velocities[idx].y = hiprand_uniform(&localState) * 2.0f - 1.0f;
        velocities[idx].z = hiprand_uniform(&localState) * 2.0f - 1.0f;
        velocities[idx].w = 0.0f; // Force accumulator
        states[idx] = localState;
    }
}

__global__ void fluidSimulationStep(float4 *positions, float4 *velocities, float deltaTime) {
    extern __shared__ float4 sharedPositions[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        sharedPositions[threadIdx.x] = positions[idx];
        __syncthreads();

        float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = 0; i < blockDim.x; ++i) {
            if (i != threadIdx.x) {
                float4 diff = make_float4(sharedPositions[i].x - sharedPositions[threadIdx.x].x,
                                          sharedPositions[i].y - sharedPositions[threadIdx.x].y,
                                          sharedPositions[i].z - sharedPositions[threadIdx.x].z, 0.0f);
                float distSquared = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                if (distSquared > 0.0f && distSquared < 1.0f) {
                    float dist = sqrtf(distSquared);
                    float3 forceDir = make_float3(diff.x / dist, diff.y / dist, diff.z / dist);
                    float forceMag = 1.0f / distSquared; // Simple repulsion force
                    force.x += forceDir.x * forceMag;
                    force.y += forceDir.y * forceMag;
                    force.z += forceDir.z * forceMag;
                }
            }
        }
        __syncthreads();

        velocities[idx].w += force.x; // Accumulate force
        positions[idx].x += velocities[idx].x * deltaTime + 0.5f * velocities[idx].w * deltaTime * deltaTime;
        positions[idx].y += velocities[idx].y * deltaTime + 0.5f * velocities[idx].w * deltaTime * deltaTime;
        positions[idx].z += velocities[idx].z * deltaTime + 0.5f * velocities[idx].w * deltaTime * deltaTime;
        velocities[idx].x += velocities[idx].w * deltaTime;
        velocities[idx].y += velocities[idx].w * deltaTime;
        velocities[idx].z += velocities[idx].w * deltaTime;
        velocities[idx].w = 0.0f; // Reset force accumulator
    }
}

int main() {
    float4 *d_positions, *d_velocities;
    hiprandState *d_states;

    hipError_t err = hipMalloc((void**)&d_positions, NUM_PARTICLES * sizeof(float4));
    if (err != hipSuccess) {
        std::cerr << "Error allocating positions: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_velocities, NUM_PARTICLES * sizeof(float4));
    if (err != hipSuccess) {
        std::cerr << "Error allocating velocities: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_states, NUM_PARTICLES * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Error allocating states: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    unsigned int seed = 1234;
    int numBlocks = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializeParticles<<<numBlocks, BLOCK_SIZE>>>(d_states, d_positions, d_velocities);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing particles: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Error synchronizing after initialization: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    float deltaTime = 0.01f;
    for (int step = 0; step < 100; ++step) {
        fluidSimulationStep<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float4)>>>(d_positions, d_velocities, deltaTime);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in simulation step " << step << ": " << hipGetErrorString(err) << std::endl;
            return -1;
        }
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::cerr << "Error synchronizing after step " << step << ": " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_states);

    return 0;
}