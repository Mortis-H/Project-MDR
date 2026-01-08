#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

// Constants for the simulation
const int N = 4096; // Number of particles
const float dt = 0.01f; // Time step
const float epsilon = 1.0f; // Lennard-Jones potential parameter
const float sigma = 1.0f; // Lennard-Jones potential parameter
const float rc = 2.5f; // Cutoff radius

// Structure to hold particle data
struct Particle {
    float4 pos; // x, y, z, w (w is unused)
    float4 vel; // vx, vy, vz, w (w is unused)
    float4 acc; // ax, ay, az, w (w is unused)
};

// Kernel to compute forces using Lennard-Jones potential
__global__ void computeForces(Particle* particles, float4* forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 pos = particles[idx].pos;

    // Loop over all other particles to compute forces
    for (int j = 0; j < N; ++j) {
        if (j == idx) continue; // Skip self-interaction

        float4 r_ji = make_float4(pos.x - particles[j].pos.x,
                                 pos.y - particles[j].pos.y,
                                 pos.z - particles[j].pos.z,
                                 0.0f);
        float r2 = r_ji.x * r_ji.x + r_ji.y * r_ji.y + r_ji.z * r_ji.z;

        if (r2 < rc * rc) {
            float r6 = r2 * r2 * r2;
            float force_factor = 24.0f * epsilon * (2.0f / (r6 * r6) - 1.0f / r6) / r2;
            force.x += force_factor * r_ji.x;
            force.y += force_factor * r_ji.y;
            force.z += force_factor * r_ji.z;
        }
    }

    forces[idx] = force;
}

// Kernel to update positions and velocities using velocity Verlet
__global__ void updatePositionsAndVelocities(Particle* particles, float4* forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float4 pos = particles[idx].pos;
    float4 vel = particles[idx].vel;
    float4 acc = particles[idx].acc;
    float4 force = forces[idx];

    // Update velocity and position
    vel.x += 0.5f * dt * (acc.x + force.x);
    vel.y += 0.5f * dt * (acc.y + force.y);
    vel.z += 0.5f * dt * (acc.z + force.z);

    pos.x += dt * vel.x;
    pos.y += dt * vel.y;
    pos.z += dt * vel.z;

    // Update acceleration
    acc.x = force.x;
    acc.y = force.y;
    acc.z = force.z;

    // Store updated values
    particles[idx].pos = pos;
    particles[idx].vel = vel;
    particles[idx].acc = acc;
}

int main() {
    // Allocate memory for particles on host
    Particle* h_particles = new Particle[N];
    float4* h_forces = new float4[N];

    // Initialize particles with random positions and zero velocities and accelerations
    for (int i = 0; i < N; ++i) {
        h_particles[i].pos = make_float4(static_cast<float>(rand()) / RAND_MAX * 10.0f,
                                         static_cast<float>(rand()) / RAND_MAX * 10.0f,
                                         static_cast<float>(rand()) / RAND_MAX * 10.0f,
                                         0.0f);
        h_particles[i].vel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        h_particles[i].acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Allocate memory for particles on device
    Particle* d_particles;
    float4* d_forces;
    hipError_t err = hipMalloc((void**)&d_particles, N * sizeof(Particle));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for particles: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMalloc((void**)&d_forces, N * sizeof(float4));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for forces: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy particles to device
    err = hipMemcpy(d_particles, h_particles, N * sizeof(Particle), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying particles to device: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Perform molecular dynamics simulation steps
    for (int step = 0; step < 100; ++step) {
        // Compute forces
        computeForces<<<gridSize, blockSize>>>(d_particles, d_forces);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in computeForces kernel: " << hipGetErrorString(err) << std::endl;
            return 1;
        }

        // Update positions and velocities
        updatePositionsAndVelocities<<<gridSize, blockSize>>>(d_particles, d_forces);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in updatePositionsAndVelocities kernel: " << hipGetErrorString(err) << std::endl;
            return 1;
        }

        // Synchronize to ensure all computations are complete
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::cerr << "Error synchronizing device: " << hipGetErrorString(err) << std::endl;
            return 1;
        }
    }

    // Copy particles back to host
    err = hipMemcpy(h_particles, d_particles, N * sizeof(Particle), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying particles back to host: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // Free device memory
    hipFree(d_particles);
    hipFree(d_forces);

    // Free host memory
    delete[] h_particles;
    delete[] h_forces;

    std::cout << "Simulation completed successfully." << std::endl;
    return 0;
}