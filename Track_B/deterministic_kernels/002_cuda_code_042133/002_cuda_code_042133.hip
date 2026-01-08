#include <iostream>
#include <hip/hip_runtime.h>


#include <hiprand/hiprand_kernel.h>

#define NUM_PARTICLES 64
#define NUM_GPUS 2
#define BLOCK_SIZE 32

// Structure to hold particle data
struct Particle {
    float4 pos;
    float4 vel;
    float4 acc;
};

// Shared memory for particle data
__shared__ Particle shared_particles[BLOCK_SIZE];

// Kernel to calculate forces using Lennard-Jones potential
__global__ void calculateForces(Particle* particles, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    Particle p = particles[idx];
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int j = 0; j < num_particles; ++j) {
        if (idx != j) {
            float4 r = make_float4(particles[j].pos.x - p.pos.x,
                                   particles[j].pos.y - p.pos.y,
                                   particles[j].pos.z - p.pos.z,
                                   0.0f);
            float r2 = r.x * r.x + r.y * r.y + r.z * r.z;
            float r6 = r2 * r2 * r2;
            float lj = 48.0f * (1.0f / r6 - 1.0f / (r6 * r6)) / r2;
            force.x += lj * r.x;
            force.y += lj * r.y;
            force.z += lj * r.z;
        }
    }
    particles[idx].acc = force;
}

// Velocity Verlet kernel
__global__ void velocityVerlet(Particle* particles, float dt, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    Particle p = particles[idx];
    p.pos.x += p.vel.x * dt + 0.5f * p.acc.x * dt * dt;
    p.pos.y += p.vel.y * dt + 0.5f * p.acc.y * dt * dt;
    p.pos.z += p.vel.z * dt + 0.5f * p.acc.z * dt * dt;

    p.vel.x += 0.5f * p.acc.x * dt;
    p.vel.y += 0.5f * p.acc.y * dt;
    p.vel.z += 0.5f * p.acc.z * dt;

    particles[idx] = p;
}

// Main function to demonstrate the usage of the kernels
int main() {
    hipSetDevice(0);
    int num_particles = NUM_PARTICLES;
    int num_particles_per_gpu = num_particles / NUM_GPUS;
    float dt = 0.001f;

    // Allocate memory on host
    Particle* h_particles = new Particle[num_particles];
    for (int i = 0; i < num_particles; ++i) {
        h_particles[i].pos = make_float4((float)i, (float)i, (float)i, 0.0f);
        h_particles[i].vel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        h_particles[i].acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Allocate memory on devices and copy data
    Particle* d_particles[NUM_GPUS];
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        hipMalloc(&d_particles[gpu], num_particles_per_gpu * sizeof(Particle));
        hipMemcpy(d_particles[gpu], h_particles + gpu * num_particles_per_gpu,
                   num_particles_per_gpu * sizeof(Particle), hipMemcpyHostToDevice);
    }

    // Launch kernels on each GPU
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        int num_blocks = (num_particles_per_gpu + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // First step: calculate forces
        calculateForces<<<num_blocks, BLOCK_SIZE>>>(d_particles[gpu], num_particles_per_gpu);
        hipDeviceSynchronize();
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in calculateForces kernel: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Second step: update positions and velocities
        velocityVerlet<<<num_blocks, BLOCK_SIZE>>>(d_particles[gpu], dt, num_particles_per_gpu);
        hipDeviceSynchronize();
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in velocityVerlet kernel: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy data back to host
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        hipMemcpy(h_particles + gpu * num_particles_per_gpu, d_particles[gpu],
                   num_particles_per_gpu * sizeof(Particle), hipMemcpyDeviceToHost);
    }

    // Free device memory
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        hipFree(d_particles[gpu]);
    }

    // Free host memory
    delete[] h_particles;

    std::cout << "Simulation completed successfully." << std::endl;
    return 0;
}