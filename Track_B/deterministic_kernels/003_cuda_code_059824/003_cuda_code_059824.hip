#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 4096
#define BLOCK_SIZE 256

// Structure to represent a particle
struct Particle {
    float x, y, z; // Position
    float vx, vy, vz; // Velocity
    float mass; // Mass
};

// Kernel to initialize particles with random positions and velocities
__global__ void initParticles(Particle *p, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        hiprandState s;
        hiprand_init((unsigned long long)clock() + idx, 0, 0, &s);
        p[idx].x = hiprand_uniform(&s) * 100.0f;
        p[idx].y = hiprand_uniform(&s) * 100.0f;
        p[idx].z = hiprand_uniform(&s) * 100.0f;
        p[idx].vx = hiprand_uniform(&s) * 1.0f - 0.5f;
        p[idx].vy = hiprand_uniform(&s) * 1.0f - 0.5f;
        p[idx].vz = hiprand_uniform(&s) * 1.0f - 0.5f;
        p[idx].mass = hiprand_uniform(&s) * 1.0f + 0.5f;
    }
}

// Kernel to compute the forces between particles
__global__ void computeForces(Particle *p, float *f, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        for (int j = 0; j < n; j++) {
            if (idx != j) {
                float dx = p[j].x - p[idx].x;
                float dy = p[j].y - p[idx].y;
                float dz = p[j].z - p[idx].z;
                float distSqr = dx * dx + dy * dy + dz * dz + 1e-10f;
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                float s = p[j].mass * invDist3;
                fx += dx * s;
                fy += dy * s;
                fz += dz * s;
            }
        }
        f[idx * 3] = fx;
        f[idx * 3 + 1] = fy;
        f[idx * 3 + 2] = fz;
    }
}

// Kernel to update particle positions and velocities
__global__ void updateParticles(Particle *p, float *f, float dt, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = f[idx * 3] / p[idx].mass;
        float ay = f[idx * 3 + 1] / p[idx].mass;
        float az = f[idx * 3 + 2] / p[idx].mass;
        p[idx].vx += ax * dt;
        p[idx].vy += ay * dt;
        p[idx].vz += az * dt;
        p[idx].x += p[idx].vx * dt;
        p[idx].y += p[idx].vy * dt;
        p[idx].z += p[idx].vz * dt;
    }
}

// Main function to demonstrate the usage of the N-body simulation
int main() {
    Particle *d_particles;
    float *d_forces;
    const float dt = 0.01f;
    const int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_particles, N * sizeof(Particle));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for particles: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_forces, N * 3 * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for forces: " << hipGetErrorString(err) << std::endl;
        hipFree(d_particles);
        return -1;
    }

    // Initialize particles
    initParticles<<<numBlocks, BLOCK_SIZE>>>(d_particles, N);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing particles: " << hipGetErrorString(err) << std::endl;
        hipFree(d_particles);
        hipFree(d_forces);
        return -1;
    }

    // Perform N-body simulation
    for (int step = 0; step < 100; step++) {
        // Compute forces
        computeForces<<<numBlocks, BLOCK_SIZE>>>(d_particles, d_forces, N);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error computing forces: " << hipGetErrorString(err) << std::endl;
            hipFree(d_particles);
            hipFree(d_forces);
            return -1;
        }

        // Update particles
        updateParticles<<<numBlocks, BLOCK_SIZE>>>(d_particles, d_forces, dt, N);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error updating particles: " << hipGetErrorString(err) << std::endl;
            hipFree(d_particles);
            hipFree(d_forces);
            return -1;
        }
    }

    // Free device memory
    hipFree(d_particles);
    hipFree(d_forces);

    return 0;
}