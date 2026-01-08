#include <iostream>
#include <hip/hip_runtime.h>
#include <cmath>

// Constants for the simulation
const int NUM_PARTICLES = 128;
const float DT = 0.005f;  // Time step
const float EPSILON = 1e-5f;  // Softening factor

// Structure to represent a particle
struct Particle {
    float4 pos;  // Position (x, y, z) and mass (w)
    float4 vel;  // Velocity (x, y, z) and padding (w)
    float4 acc;  // Acceleration (x, y, z) and padding (w)
};

// CUDA kernel to compute the acceleration of each particle
__global__ void computeAcceleration(Particle* particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 pi = particles[idx].pos;

    for (int j = 0; j < NUM_PARTICLES; j++) {
        if (idx == j) continue;

        float4 pj = particles[j].pos;
        float4 r = make_float4(pj.x - pi.x, pj.y - pi.y, pj.z - pi.z, 0.0f);
        float rSquared = r.x * r.x + r.y * r.y + r.z * r.z + EPSILON * EPSILON;
        float invR = rsqrtf(rSquared);
        float invR3 = invR * invR * invR;
        float force = pj.w * invR3;

        acc.x += force * r.x;
        acc.y += force * r.y;
        acc.z += force * r.z;
    }

    particles[idx].acc = acc;
}

// CUDA kernel to update positions and velocities using leapfrog integration
__global__ void updateParticles(Particle* particles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    float4 pos = particles[idx].pos;
    float4 vel = particles[idx].vel;
    float4 acc = particles[idx].acc;

    // Update position
    pos.x += vel.x * dt + 0.5f * acc.x * dt * dt;
    pos.y += vel.y * dt + 0.5f * acc.y * dt * dt;
    pos.z += vel.z * dt + 0.5f * acc.z * dt * dt;

    // Update velocity
    vel.x += acc.x * dt;
    vel.y += acc.y * dt;
    vel.z += acc.z * dt;

    particles[idx].pos = pos;
    particles[idx].vel = vel;
}

int main() {
    // Initialize particles
    Particle* particles;
    hipMallocManaged(&particles, NUM_PARTICLES * sizeof(Particle));

    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].pos = make_float4((float)i, (float)i, (float)i, 1.0f);  // Simple initialization
        particles[i].vel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        particles[i].acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;

    // Perform a few time steps of the simulation
    for (int step = 0; step < 100; step++) {
        // Compute acceleration
        computeAcceleration<<<blocksPerGrid, threadsPerBlock>>>(particles);
        hipDeviceSynchronize();

        // Update positions and velocities
        updateParticles<<<blocksPerGrid, threadsPerBlock>>>(particles, DT);
        hipDeviceSynchronize();

        // Check for errors
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Free memory
    hipFree(particles);

    std::cout << "Simulation completed successfully." << std::endl;
    return 0;
}