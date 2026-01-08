#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 16384
#define G 0.0000000000667430f // Gravitational constant in m^3 kg^-1 s^-2
#define DT 0.01f // Time step in seconds
#define SOFTENING 0.00001f // Softening factor to prevent division by zero

// Structure to hold particle data
struct Particle {
    float4 pos; // x, y, z, mass
    float4 vel; // vx, vy, vz, unused
};

// CUDA kernel to perform N-body simulation
__global__ void nbodyKernel(Particle* particles, Particle* newParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 pos = particles[idx].pos;
    float mass = pos.w;

    // Calculate the net force acting on each particle
    for (int j = 0; j < N; j++) {
        if (j == idx) continue;

        float4 posJ = particles[j].pos;
        float dx = posJ.x - pos.x;
        float dy = posJ.y - pos.y;
        float dz = posJ.z - pos.z;
        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING * SOFTENING;
        float dist = rsqrtf(distSqr);
        float invDistCube = dist * dist * dist;
        float s = G * posJ.w * invDistCube;

        acc.x += dx * s;
        acc.y += dy * s;
        acc.z += dz * s;
    }

    // Update position and velocity
    float4 vel = particles[idx].vel;
    vel.x += acc.x * DT;
    vel.y += acc.y * DT;
    vel.z += acc.z * DT;

    pos.x += vel.x * DT;
    pos.y += vel.y * DT;
    pos.z += vel.z * DT;

    // Store the updated position and velocity
    newParticles[idx].pos = pos;
    newParticles[idx].vel = vel;
}

// Function to initialize particles with random positions and velocities
__global__ void initParticles(Particle* particles, hiprandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Initialize curand state
    hiprandState localState = states[idx];
    hiprand_init((unsigned long long) clock() + idx, 0, 0, &localState);

    // Generate random position and velocity
    particles[idx].pos.x = hiprand_uniform(&localState) * 100.0f;
    particles[idx].pos.y = hiprand_uniform(&localState) * 100.0f;
    particles[idx].pos.z = hiprand_uniform(&localState) * 100.0f;
    particles[idx].pos.w = 1.0f + hiprand_uniform(&localState) * 10.0f; // Mass

    particles[idx].vel.x = hiprand_uniform(&localState) * 0.1f - 0.05f;
    particles[idx].vel.y = hiprand_uniform(&localState) * 0.1f - 0.05f;
    particles[idx].vel.z = hiprand_uniform(&localState) * 0.1f - 0.05f;

    // Store the state
    states[idx] = localState;
}

int main() {
    // Allocate memory for particles on host
    Particle* h_particles = new Particle[N];
    Particle* h_newParticles = new Particle[N];

    // Allocate memory for particles on device
    Particle* d_particles;
    Particle* d_newParticles;
    hipMalloc((void**)&d_particles, N * sizeof(Particle));
    hipMalloc((void**)&d_newParticles, N * sizeof(Particle));

    // Allocate and initialize random states on device
    hiprandState* d_states;
    hipMalloc((void**)&d_states, N * sizeof(hiprandState));
    initParticles<<<(N + 255) / 256, 256>>>(d_particles, d_states);
    hipDeviceSynchronize();

    // Check for errors in memory allocation and initialization
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Perform N-body simulation
    nbodyKernel<<<(N + 255) / 256, 256>>>(d_particles, d_newParticles);
    hipDeviceSynchronize();

    // Check for errors in kernel execution
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    hipMemcpy(h_newParticles, d_newParticles, N * sizeof(Particle), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_particles);
    hipFree(d_newParticles);
    hipFree(d_states);

    // Free host memory
    delete[] h_particles;
    delete[] h_newParticles;

    return 0;
}