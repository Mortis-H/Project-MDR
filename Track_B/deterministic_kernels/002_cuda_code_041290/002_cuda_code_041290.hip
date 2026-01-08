#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <vector>

#define N 1024
#define THETA 0.5f
#define BLOCK_SIZE 256

// Structure to hold particle data
struct Particle {
    float4 pos; // x, y, z, mass
    float4 acc; // ax, ay, az, unused
};

// Kernel to calculate the gravitational force on each particle
__global__ void nbody_kernel(Particle* particles, Particle* tempParticles, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    Particle p = particles[idx];
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int j = 0; j < numParticles; j++) {
        if (j == idx) continue;
        float4 q = particles[j].pos;
        float4 r = make_float4(q.x - p.pos.x, q.y - p.pos.y, q.z - p.pos.z, 0.0f);
        float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
        float dist = sqrtf(distSqr);
        if (dist > 0.0f) {
            float invDistCube = 1.0f / (distSqr * dist);
            float s = p.pos.w * q.w * invDistCube;
            force.x += s * r.x;
            force.y += s * r.y;
            force.z += s * r.z;
        }
    }
    tempParticles[idx].acc = force;
}

// Kernel to update particle positions and velocities
__global__ void update_kernel(Particle* particles, float dt, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    Particle p = particles[idx];
    float4 a = p.acc;
    p.pos.x += p.acc.x * dt * dt;
    p.pos.y += p.acc.y * dt * dt;
    p.pos.z += p.acc.z * dt * dt;
    particles[idx] = p;
}

// Multi-GPU setup and kernel launch
void simulate_nbody(Particle* particles, int numParticles, float dt, int steps) {
    int numDevices;
    hipGetDeviceCount(&numDevices);
    if (numDevices < 2) {
        std::cerr << "This example requires at least 2 GPUs." << std::endl;
        return;
    }

    int particlesPerDevice = numParticles / numDevices;
    Particle* d_particles[numDevices];
    Particle* d_tempParticles[numDevices];

    for (int dev = 0; dev < numDevices; dev++) {
        hipSetDevice(dev);
        hipMalloc(&d_particles[dev], particlesPerDevice * sizeof(Particle));
        hipMalloc(&d_tempParticles[dev], particlesPerDevice * sizeof(Particle));
        hipMemcpy(d_particles[dev], particles + dev * particlesPerDevice, particlesPerDevice * sizeof(Particle), hipMemcpyHostToDevice);
    }

    for (int step = 0; step < steps; step++) {
        for (int dev = 0; dev < numDevices; dev++) {
            hipSetDevice(dev);
            nbody_kernel<<<(particlesPerDevice + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles[dev], d_tempParticles[dev], particlesPerDevice);
            update_kernel<<<(particlesPerDevice + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles[dev], dt, particlesPerDevice);
        }
        hipDeviceSynchronize();
    }

    for (int dev = 0; dev < numDevices; dev++) {
        hipSetDevice(dev);
        hipMemcpy(particles + dev * particlesPerDevice, d_particles[dev], particlesPerDevice * sizeof(Particle), hipMemcpyDeviceToHost);
        hipFree(d_particles[dev]);
        hipFree(d_tempParticles[dev]);
    }
}

int main() {
    std::vector<Particle> particles(N);
    for (int i = 0; i < N; i++) {
        particles[i].pos = make_float4(static_cast<float>(i), static_cast<float>(i), static_cast<float>(i), 1.0f);
        particles[i].acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float dt = 0.01f;
    int steps = 10;

    simulate_nbody(particles.data(), N, dt, steps);

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Simulation completed successfully." << std::endl;
    return 0;
}