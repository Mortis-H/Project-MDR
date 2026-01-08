#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 256
#define BLOCK_SIZE 16

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};

__global__ void initializeParticles(Particle *particles, hiprandState *states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        hiprandState localState = states[idx];
        hiprand_init(seed, idx, 0, &localState);
        particles[idx].x = hiprand_uniform(&localState) * 100.0f;
        particles[idx].y = hiprand_uniform(&localState) * 100.0f;
        particles[idx].z = hiprand_uniform(&localState) * 100.0f;
        particles[idx].vx = hiprand_uniform(&localState) * 1.0f - 0.5f;
        particles[idx].vy = hiprand_uniform(&localState) * 1.0f - 0.5f;
        particles[idx].vz = hiprand_uniform(&localState) * 1.0f - 0.5f;
        particles[idx].mass = hiprand_uniform(&localState) * 1.0f + 0.1f;
        states[idx] = localState;
    }
}

__device__ float3 calculateForce(Particle p1, Particle p2) {
    float3 force = {0.0f, 0.0f, 0.0f};
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float dz = p2.z - p1.z;
    float distSqr = dx * dx + dy * dy + dz * dz + 0.0001f; // Softening factor
    float dist = sqrtf(distSqr);
    float F = p1.mass * p2.mass / distSqr;
    force.x = F * dx / dist;
    force.y = F * dy / dist;
    force.z = F * dz / dist;
    return force;
}

__global__ void nbodySimulation(Particle *particles, float dt) {
    extern __shared__ Particle sharedParticles[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    if (idx < N) {
        sharedParticles[tid] = particles[idx];
    }
    __syncthreads();

    float3 force = {0.0f, 0.0f, 0.0f};

    // Calculate forces
    for (int i = 0; i < blockDim.x; ++i) {
        if (tid != i) {
            force.x += calculateForce(sharedParticles[tid], sharedParticles[i]).x;
            force.y += calculateForce(sharedParticles[tid], sharedParticles[i]).y;
            force.z += calculateForce(sharedParticles[tid], sharedParticles[i]).z;
        }
    }
    __syncthreads();

    // Update velocities and positions
    if (idx < N) {
        particles[idx].vx += force.x * dt / particles[idx].mass;
        particles[idx].vy += force.y * dt / particles[idx].mass;
        particles[idx].vz += force.z * dt / particles[idx].mass;

        particles[idx].x += particles[idx].vx * dt;
        particles[idx].y += particles[idx].vy * dt;
        particles[idx].z += particles[idx].vz * dt;
    }
}

int main() {
    Particle *d_particles;
    hiprandState *d_states;
    Particle h_particles[N];

    hipError_t err = hipMalloc((void**)&d_particles, N * sizeof(Particle));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for particles" << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_states, N * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for states" << std::endl;
        hipFree(d_particles);
        return -1;
    }

    initializeParticles<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles, d_states, 1234);
    hipDeviceSynchronize();

    for (int step = 0; step < 100; ++step) {
        nbodySimulation<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(Particle)>>>(d_particles, 0.01f);
        hipDeviceSynchronize();
    }

    err = hipMemcpy(h_particles, d_particles, N * sizeof(Particle), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy particles from device to host" << std::endl;
        hipFree(d_particles);
        hipFree(d_states);
        return -1;
    }

    // Print the first few particles' positions
    for (int i = 0; i < 5; ++i) {
        std::cout << "Particle " << i << ": (" << h_particles[i].x << ", " << h_particles[i].y << ", " << h_particles[i].z << ")" << std::endl;
    }

    hipFree(d_particles);
    hipFree(d_states);
    return 0;
}