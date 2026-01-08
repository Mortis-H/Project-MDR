#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARTICLES 64
#define DT 0.001f
#define CUTOFF 1.0f
#define EPSILON 1.0f
#define SIGMA 1.0f

__global__ void initRandomStates(hiprandState *states, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < NUM_PARTICLES) {
        hiprand_init(seed, id, 0, &states[id]);
    }
}

__global__ void initPositions(hiprandState *states, float4 *positions) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < NUM_PARTICLES) {
        positions[id].x = hiprand_uniform(&states[id]) * 10.0f;
        positions[id].y = hiprand_uniform(&states[id]) * 10.0f;
        positions[id].z = hiprand_uniform(&states[id]) * 10.0f;
        positions[id].w = 0.0f; // Mass
    }
}

__global__ void initVelocities(hiprandState *states, float4 *velocities) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < NUM_PARTICLES) {
        velocities[id].x = hiprand_normal(&states[id]) * 0.1f;
        velocities[id].y = hiprand_normal(&states[id]) * 0.1f;
        velocities[id].z = hiprand_normal(&states[id]) * 0.1f;
        velocities[id].w = 0.0f; // Zero acceleration
    }
}

__device__ float4 computeForce(float4 pos1, float4 pos2) {
    float4 force = {0.0f, 0.0f, 0.0f, 0.0f};
    float dx = pos2.x - pos1.x;
    float dy = pos2.y - pos1.y;
    float dz = pos2.z - pos1.z;
    float r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < CUTOFF * CUTOFF && r2 > 0.0f) {
        float r6 = r2 * r2 * r2;
        float lj = 24.0f * EPSILON * ((2.0f * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA / r6 / r6) - (SIGMA * SIGMA * SIGMA * SIGMA / r6));
        force.x = lj * dx;
        force.y = lj * dy;
        force.z = lj * dz;
    }
    return force;
}

__global__ void velocityVerletStep1(float4 *positions, float4 *velocities, float4 *forces, float dt) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < NUM_PARTICLES) {
        positions[id].x += velocities[id].x * dt + 0.5f * forces[id].x * dt * dt;
        positions[id].y += velocities[id].y * dt + 0.5f * forces[id].y * dt * dt;
        positions[id].z += velocities[id].z * dt + 0.5f * forces[id].z * dt * dt;
    }
}

__global__ void computeForces(float4 *positions, float4 *forces) {
    extern __shared__ float4 shared_positions[];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    shared_positions[tid] = positions[id];
    __syncthreads();

    float4 force = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = 0; j < blockDim.x; ++j) {
        if (tid != j) {
            force = computeForce(shared_positions[tid], shared_positions[j]);
            atomicAdd(&forces[id].x, force.x);
            atomicAdd(&forces[id].y, force.y);
            atomicAdd(&forces[id].z, force.z);
        }
    }
    __syncthreads();
}

__global__ void velocityVerletStep2(float4 *velocities, float4 *forces, float4 *old_forces, float dt) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < NUM_PARTICLES) {
        velocities[id].x += 0.5f * (forces[id].x + old_forces[id].x) * dt;
        velocities[id].y += 0.5f * (forces[id].y + old_forces[id].y) * dt;
        velocities[id].z += 0.5f * (forces[id].z + old_forces[id].z) * dt;
    }
}

int main() {
    float4 *d_positions, *d_velocities, *d_forces, *d_old_forces;
    hiprandState *d_states;

    hipError_t err;

    err = hipMalloc(&d_positions, NUM_PARTICLES * sizeof(float4));
    if (err != hipSuccess) { std::cerr << "Error allocating positions: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_velocities, NUM_PARTICLES * sizeof(float4));
    if (err != hipSuccess) { std::cerr << "Error allocating velocities: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_forces, NUM_PARTICLES * sizeof(float4));
    if (err != hipSuccess) { std::cerr << "Error allocating forces: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_old_forces, NUM_PARTICLES * sizeof(float4));
    if (err != hipSuccess) { std::cerr << "Error allocating old forces: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_states, NUM_PARTICLES * sizeof(hiprandState));
    if (err != hipSuccess) { std::cerr << "Error allocating states: " << hipGetErrorString(err) << std::endl; return -1; }

    int blockSize = 32;
    int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;

    initRandomStates<<<numBlocks, blockSize>>>(d_states, time(NULL));
    hipDeviceSynchronize();

    initPositions<<<numBlocks, blockSize>>>(d_states, d_positions);
    hipDeviceSynchronize();

    initVelocities<<<numBlocks, blockSize>>>(d_states, d_velocities);
    hipDeviceSynchronize();

    hipMemset(d_forces, 0, NUM_PARTICLES * sizeof(float4));
    hipMemset(d_old_forces, 0, NUM_PARTICLES * sizeof(float4));

    for (int step = 0; step < 1000; ++step) {
        // Step 1: Update positions
        velocityVerletStep1<<<numBlocks, blockSize>>>(d_positions, d_velocities, d_forces, DT);

        // Step 2: Compute forces
        computeForces<<<numBlocks, blockSize, blockSize * sizeof(float4)>>>(d_positions, d_forces);

        // Step 3: Update velocities
        velocityVerletStep2<<<numBlocks, blockSize>>>(d_velocities, d_forces, d_old_forces, DT);

        // Swap forces
        float4 *temp = d_old_forces;
        d_old_forces = d_forces;
        d_forces = temp;
    }

    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_forces);
    hipFree(d_old_forces);
    hipFree(d_states);

    return 0;
}