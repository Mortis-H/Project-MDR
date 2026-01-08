#include <iostream>
#include <hip/hip_runtime.h>
#include <vector>

// Constants
const int N = 64;          // Number of particles
const float G = 6.673e-11f; // Gravitational constant
const float dt = 0.01f;    // Time step

// Structure to represent a particle
struct Particle {
    float4 pos; // Position (x, y, z, w=mass)
    float4 vel; // Velocity (vx, vy, vz, w=0)
};

// Kernel to compute forces between particles using brute force
__global__ void computeForces(Particle* particles, Particle* forces) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        float4 posA = particles[idx].pos;
        float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (int j = 0; j < N; ++j) {
            if (idx != j) {
                float4 posB = particles[j].pos;
                float3 r = make_float3(posB.x - posA.x, posB.y - posA.y, posB.z - posA.z);
                float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 1e-12f; // Softening factor
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                float s = G * posA.w * posB.w * invDist3;

                force.x += s * r.x;
                force.y += s * r.y;
                force.z += s * r.z;
            }
        }
        forces[idx].vel = force;
    }
}

// Kernel to update positions and velocities
__global__ void updateParticles(Particle* particles, Particle* forces) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        float4 pos = particles[idx].pos;
        float4 vel = particles[idx].vel;
        float4 force = forces[idx].vel;

        // Update velocity
        vel.x += force.x * dt / pos.w;
        vel.y += force.y * dt / pos.w;
        vel.z += force.z * dt / pos.w;

        // Update position
        pos.x += vel.x * dt;
        pos.y += vel.y * dt;
        pos.z += vel.z * dt;

        particles[idx].pos = pos;
        particles[idx].vel = vel;
    }
}

// Error checking macro
#define cudaCheckError() {                                          \
    hipError_t e = hipGetLastError();                             \
    if (e != hipSuccess) {                                         \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

int main() {
    // Initialize host particles
    std::vector<Particle> h_particles(N);
    std::vector<Particle> h_forces(N);
    for (int i = 0; i < N; ++i) {
        h_particles[i].pos = make_float4(i, i, i, 1.0f + i % 10); // Random initial positions and masses
        h_particles[i].vel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Allocate device memory
    Particle* d_particles;
    Particle* d_forces;
    hipMalloc(&d_particles, N * sizeof(Particle));
    hipMalloc(&d_forces, N * sizeof(Particle));
    cudaCheckError();

    // Copy particles to device
    hipMemcpy(d_particles, h_particles.data(), N * sizeof(Particle), hipMemcpyHostToDevice);
    cudaCheckError();

    // Define grid and block dimensions
    dim3 threadsPerBlock(256);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Launch kernels
    computeForces<<<numBlocks, threadsPerBlock>>>(d_particles, d_forces);
    cudaCheckError();

    updateParticles<<<numBlocks, threadsPerBlock>>>(d_particles, d_forces);
    cudaCheckError();

    // Copy results back to host
    hipMemcpy(h_particles.data(), d_particles, N * sizeof(Particle), hipMemcpyDeviceToHost);
    cudaCheckError();

    // Free device memory
    hipFree(d_particles);
    hipFree(d_forces);
    cudaCheckError();

    // Output final positions (for demonstration)
    for (int i = 0; i < N; ++i) {
        std::cout << "Particle " << i << ": (" << h_particles[i].pos.x << ", " << h_particles[i].pos.y << ", " << h_particles[i].pos.z << ")\n";
    }

    return 0;
}