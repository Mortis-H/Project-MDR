#include <iostream>
#include <hip/hip_runtime.h>


// Constants for the simulation
#define NUM_PARTICLES 256
#define DIM 3
#define DT 0.001f
#define EPS 1e-5f

// Structure to hold particle data
struct Particle {
    float r[DIM]; // Position
    float v[DIM]; // Velocity
    float a[DIM]; // Acceleration
    float a_prev[DIM]; // Previous acceleration
};

// Kernel to calculate forces (simple Lennard-Jones potential for demonstration)
__global__ void calculateForces(Particle *particles, float *forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    float force[DIM] = {0.0f, 0.0f, 0.0f};
    for (int j = 0; j < NUM_PARTICLES; ++j) {
        if (j == idx) continue;

        float r[DIM] = {0.0f, 0.0f, 0.0f};
        for (int d = 0; d < DIM; ++d) {
            r[d] = particles[idx].r[d] - particles[j].r[d];
        }

        float r2 = 0.0f;
        for (int d = 0; d < DIM; ++d) {
            r2 += r[d] * r[d];
        }

        if (r2 < EPS) continue;

        float r6 = r2 * r2 * r2;
        float fac = 24.0f * (2.0f / r6 - 1.0f) / r2 / r6;

        for (int d = 0; d < DIM; ++d) {
            force[d] += fac * r[d];
        }
    }

    for (int d = 0; d < DIM; ++d) {
        forces[idx * DIM + d] = force[d];
    }
}

// Kernel to update particle positions and velocities using Beeman's algorithm
__global__ void updateParticles(Particle *particles, float *forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    float a_new[DIM];
    for (int d = 0; d < DIM; ++d) {
        a_new[d] = forces[idx * DIM + d];
    }

    for (int d = 0; d < DIM; ++d) {
        particles[idx].r[d] += particles[idx].v[d] * DT + 
                              (4.0f / 3.0f * particles[idx].a[d] - 
                               1.0f / 3.0f * particles[idx].a_prev[d]) * DT * DT;
        particles[idx].v[d] += (3.0f / 2.0f * a_new[d] - 
                               1.0f / 2.0f * particles[idx].a_prev[d]) * DT;
        particles[idx].a_prev[d] = particles[idx].a[d];
        particles[idx].a[d] = a_new[d];
    }
}

// Main function to run the simulation
int main() {
    Particle *h_particles = new Particle[NUM_PARTICLES];
    Particle *d_particles;
    float *d_forces;

    // Initialize particles (simple initialization for demonstration)
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        for (int d = 0; d < DIM; ++d) {
            h_particles[i].r[d] = static_cast<float>(i % 16) + d * 0.1f;
            h_particles[i].v[d] = 0.0f;
            h_particles[i].a[d] = 0.0f;
            h_particles[i].a_prev[d] = 0.0f;
        }
    }

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_particles, NUM_PARTICLES * sizeof(Particle));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for particles: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_forces, NUM_PARTICLES * DIM * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for forces: " << hipGetErrorString(err) << std::endl;
        hipFree(d_particles);
        return -1;
    }

    // Copy particles to the device
    err = hipMemcpy(d_particles, h_particles, NUM_PARTICLES * sizeof(Particle), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy particles to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_particles);
        hipFree(d_forces);
        return -1;
    }

    // Define grid and block dimensions
    dim3 blockSize(256);
    dim3 gridSize((NUM_PARTICLES + blockSize.x - 1) / blockSize.x);

    // Run the simulation for a number of steps
    int numSteps = 100;
    for (int step = 0; step < numSteps; ++step) {
        calculateForces<<<gridSize, blockSize>>>(d_particles, d_forces);
        hipDeviceSynchronize();
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in calculateForces kernel: " << hipGetErrorString(err) << std::endl;
            break;
        }

        updateParticles<<<gridSize, blockSize>>>(d_particles, d_forces);
        hipDeviceSynchronize();
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in updateParticles kernel: " << hipGetErrorString(err) << std::endl;
            break;
        }
    }

    // Copy particles back to the host
    err = hipMemcpy(h_particles, d_particles, NUM_PARTICLES * sizeof(Particle), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy particles back to host: " << hipGetErrorString(err) << std::endl;
    }

    // Free device memory
    hipFree(d_particles);
    hipFree(d_forces);

    // Free host memory
    delete[] h_particles;

    return 0;
}