#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 64
#define BLOCK_SIZE 16

// Define a simple 3D vector structure
struct Vec3 {
    float x, y, z;
};

// Define a particle structure
struct Particle {
    Vec3 pos;
    Vec3 vel;
    float mass;
};

// Kernel to compute the forces between particles
__global__ void computeForces(Particle* particles, Vec3* forces, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    Vec3 f = {0.0f, 0.0f, 0.0f};

    for (int j = 0; j < N; j++) {
        if (i == j) continue;

        Vec3 r = {
            particles[j].pos.x - particles[i].pos.x,
            particles[j].pos.y - particles[i].pos.y,
            particles[j].pos.z - particles[i].pos.z
        };

        float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 1e-6f; // Softening factor
        float dist = sqrtf(distSqr);
        float invDist3 = 1.0f / (distSqr * dist);

        f.x += r.x * invDist3 * particles[j].mass;
        f.y += r.y * invDist3 * particles[j].mass;
        f.z += r.z * invDist3 * particles[j].mass;
    }

    forces[i] = f;
}

// Kernel to update particle positions and velocities
__global__ void updateParticles(Particle* particles, Vec3* forces, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float invMass = 1.0f / particles[i].mass;

    particles[i].vel.x += forces[i].x * invMass * dt;
    particles[i].vel.y += forces[i].y * invMass * dt;
    particles[i].vel.z += forces[i].z * invMass * dt;

    particles[i].pos.x += particles[i].vel.x * dt;
    particles[i].pos.y += particles[i].vel.y * dt;
    particles[i].pos.z += particles[i].vel.z * dt;
}

// Initialize particles with random positions and velocities
__global__ void initializeParticles(Particle* particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    hiprandState state;
    hiprand_init((unsigned long long) clock() + i, 0, 0, &state);

    particles[i].pos.x = hiprand_uniform(&state) * 100.0f - 50.0f;
    particles[i].pos.y = hiprand_uniform(&state) * 100.0f - 50.0f;
    particles[i].pos.z = hiprand_uniform(&state) * 100.0f - 50.0f;

    particles[i].vel.x = hiprand_uniform(&state) * 2.0f - 1.0f;
    particles[i].vel.y = hiprand_uniform(&state) * 2.0f - 1.0f;
    particles[i].vel.z = hiprand_uniform(&state) * 2.0f - 1.0f;

    particles[i].mass = 1.0f;
}

int main() {
    Particle* d_particles;
    Vec3* d_forces;
    Particle* h_particles = new Particle[N];

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_particles, N * sizeof(Particle));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for particles: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_forces, N * sizeof(Vec3));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for forces: " << hipGetErrorString(err) << std::endl;
        hipFree(d_particles);
        return -1;
    }

    // Initialize particles on the device
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializeParticles<<<numBlocks, BLOCK_SIZE>>>(d_particles);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing particles: " << hipGetErrorString(err) << std::endl;
        hipFree(d_particles);
        hipFree(d_forces);
        return -1;
    }

    // Perform N-body simulation
    float dt = 0.01f;
    for (int step = 0; step < 100; step++) {
        computeForces<<<numBlocks, BLOCK_SIZE>>>(d_particles, d_forces, dt);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error computing forces: " << hipGetErrorString(err) << std::endl;
            break;
        }

        updateParticles<<<numBlocks, BLOCK_SIZE>>>(d_particles, d_forces, dt);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error updating particles: " << hipGetErrorString(err) << std::endl;
            break;
        }
    }

    // Copy results back to host
    err = hipMemcpy(h_particles, d_particles, N * sizeof(Particle), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying particles from device to host: " << hipGetErrorString(err) << std::endl;
    }

    // Free device memory
    hipFree(d_particles);
    hipFree(d_forces);

    // Free host memory
    delete[] h_particles;

    return 0;
}