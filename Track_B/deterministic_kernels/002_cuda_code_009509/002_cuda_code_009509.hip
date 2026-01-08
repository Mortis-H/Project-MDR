#include <iostream>
#include <hip/hip_runtime.h>

// Constants for the simulation
#define NUM_PARTICLES 64
#define DT 0.001f
#define EPSILON 1e-5f
#define G 6.67430e-11f

// Structure to hold particle data
struct Particle {
    float4 pos;  // Position (x, y, z, w)
    float4 vel;  // Velocity (vx, vy, vz, w)
    float4 acc;  // Acceleration (ax, ay, az, w)
    float4 acc1; // Acceleration at t-dt (ax1, ay1, az1, w)
};

// Kernel to compute forces using a simple gravitational force model
__global__ void computeForces(Particle* particles, Particle* particles_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 pos = particles[idx].pos;

    for (int j = 0; j < NUM_PARTICLES; ++j) {
        if (j != idx) {
            float4 pos_j = particles[j].pos;
            float4 r = make_float4(pos_j.x - pos.x, pos_j.y - pos.y, pos_j.z - pos.z, 0.0f);
            float rsq = r.x * r.x + r.y * r.y + r.z * r.z + EPSILON;
            float inv_r = rsq == 0.0f ? 0.0f : rsqrtf(rsq);
            float inv_r3 = inv_r * inv_r * inv_r;
            float f = G * inv_r3;

            atomicAdd(&particles_out[idx].acc.x, f * r.x);
            atomicAdd(&particles_out[idx].acc.y, f * r.y);
            atomicAdd(&particles_out[idx].acc.z, f * r.z);
        }
    }
}

// Kernel to update positions and velocities using Beeman's algorithm
__global__ void updateParticles(Particle* particles, Particle* particles_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    float4 pos = particles[idx].pos;
    float4 vel = particles[idx].vel;
    float4 acc = particles[idx].acc;
    float4 acc1 = particles[idx].acc1;

    // Update position
    float4 pos_new = make_float4(
        pos.x + vel.x * DT + (4.0f / 3.0f * acc.x - 1.0f / 3.0f * acc1.x) * DT * DT,
        pos.y + vel.y * DT + (4.0f / 3.0f * acc.y - 1.0f / 3.0f * acc1.y) * DT * DT,
        pos.z + vel.z * DT + (4.0f / 3.0f * acc.z - 1.0f / 3.0f * acc1.z) * DT * DT,
        0.0f);

    // Update velocity
    float4 vel_new = make_float4(
        vel.x + (3.0f / 2.0f * acc.x - 1.0f / 2.0f * acc1.x) * DT,
        vel.y + (3.0f / 2.0f * acc.y - 1.0f / 2.0f * acc1.y) * DT,
        vel.z + (3.0f / 2.0f * acc.z - 1.0f / 2.0f * acc1.z) * DT,
        0.0f);

    // Store new position, velocity, and old acceleration
    particles_out[idx].pos = pos_new;
    particles_out[idx].vel = vel_new;
    particles_out[idx].acc1 = acc;
}

int main() {
    // Allocate memory for particles on host
    Particle* h_particles = new Particle[NUM_PARTICLES];
    Particle* h_particles_out = new Particle[NUM_PARTICLES];

    // Initialize particles (simple initialization)
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_particles[i].pos = make_float4(i * 1.0f, 0.0f, 0.0f, 0.0f);
        h_particles[i].vel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        h_particles[i].acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        h_particles[i].acc1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Allocate memory for particles on device
    Particle* d_particles;
    Particle* d_particles_out;
    hipMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle));
    hipMalloc(&d_particles_out, NUM_PARTICLES * sizeof(Particle));

    // Copy particles to device
    hipMemcpy(d_particles, h_particles, NUM_PARTICLES * sizeof(Particle), hipMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16);
    dim3 gridSize((NUM_PARTICLES + blockSize.x - 1) / blockSize.x);

    // Perform a single step of molecular dynamics simulation
    computeForces<<<gridSize, blockSize>>>(d_particles, d_particles_out);
    hipDeviceSynchronize(); // Ensure forces are computed before updating

    updateParticles<<<gridSize, blockSize>>>(d_particles_out, d_particles);
    hipDeviceSynchronize(); // Ensure particles are updated before copying back

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy updated particles back to host
    hipMemcpy(h_particles_out, d_particles, NUM_PARTICLES * sizeof(Particle), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_particles);
    hipFree(d_particles_out);

    // Free host memory
    delete[] h_particles;
    delete[] h_particles_out;

    return 0;
}