#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define N 4096
#define BLOCK_SIZE 256

// Simple structure to hold particle data
struct Particle {
    float4 pos; // x, y, z, mass
    float4 acc; // ax, ay, az, (padding)
};

// CUDA kernel to calculate forces between particles
__global__ void calculateForces(Particle *p, float3 *forces) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float3 f = {0.0f, 0.0f, 0.0f};
        for (int j = 0; j < N; j++) {
            if (i != j) {
                float3 r = {p[j].pos.x - p[i].pos.x, p[j].pos.y - p[i].pos.y, p[j].pos.z - p[i].pos.z};
                float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 1e-10f; // prevent division by zero
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                f.x += r.x * invDist3 * p[j].pos.w;
                f.y += r.y * invDist3 * p[j].pos.w;
                f.z += r.z * invDist3 * p[j].pos.w;
            }
        }
        forces[i] = f;
    }
}

// CUDA kernel to update particle positions
__global__ void updatePositions(Particle *p, float3 *forces, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        p[i].acc.x = forces[i].x / p[i].pos.w;
        p[i].acc.y = forces[i].y / p[i].pos.w;
        p[i].acc.z = forces[i].z / p[i].pos.w;

        p[i].pos.x += p[i].acc.x * dt * dt;
        p[i].pos.y += p[i].acc.y * dt * dt;
        p[i].pos.z += p[i].acc.z * dt * dt;
    }
}

// Main function to run the simulation
int main() {
    Particle *d_particles;
    float3 *d_forces;
    Particle h_particles[N];

    // Initialize particles on host
    for (int i = 0; i < N; i++) {
        h_particles[i].pos.x = (float(rand()) / RAND_MAX) * 100.0f;
        h_particles[i].pos.y = (float(rand()) / RAND_MAX) * 100.0f;
        h_particles[i].pos.z = (float(rand()) / RAND_MAX) * 100.0f;
        h_particles[i].pos.w = 1.0f; // mass
        h_particles[i].acc.x = 0.0f;
        h_particles[i].acc.y = 0.0f;
        h_particles[i].acc.z = 0.0f;
    }

    // Allocate memory on device
    hipError_t err = hipMalloc((void **)&d_particles, N * sizeof(Particle));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for particles (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_forces, N * sizeof(float3));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for forces (error code " << err << ")" << std::endl;
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_particles, h_particles, N * sizeof(Particle), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from host to device (error code " << err << ")" << std::endl;
        return -1;
    }

    // Simulation parameters
    float dt = 0.01f;
    int numSteps = 100;

    // Run the simulation
    for (int step = 0; step < numSteps; step++) {
        calculateForces<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles, d_forces);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Kernel launch failed (error code " << err << ")" << std::endl;
            return -1;
        }

        updatePositions<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles, d_forces, dt);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Kernel launch failed (error code " << err << ")" << std::endl;
            return -1;
        }

        hipDeviceSynchronize();
    }

    // Copy data back to host
    err = hipMemcpy(h_particles, d_particles, N * sizeof(Particle), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device to host (error code " << err << ")" << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_particles);
    hipFree(d_forces);

    std::cout << "Simulation completed successfully." << std::endl;
    return 0;
}