#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARTICLES 512
#define BLOCK_SIZE 256
#define GRID_SIZE (NUM_PARTICLES / BLOCK_SIZE)

// Define a simple particle structure
struct Particle {
    float4 pos;  // Position (x, y, z, w)
    float4 vel;  // Velocity (vx, vy, vz, w)
};

// Kernel to initialize particles with random positions and velocities
__global__ void initParticles(Particle *particles, hiprandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        hiprandState localState = states[idx];
        particles[idx].pos.x = hiprand_uniform(&localState) * 100.0f;
        particles[idx].pos.y = hiprand_uniform(&localState) * 100.0f;
        particles[idx].pos.z = hiprand_uniform(&localState) * 100.0f;
        particles[idx].pos.w = 0.0f;
        particles[idx].vel.x = hiprand_uniform(&localState) * 1.0f - 0.5f;
        particles[idx].vel.y = hiprand_uniform(&localState) * 1.0f - 0.5f;
        particles[idx].vel.z = hiprand_uniform(&localState) * 1.0f - 0.5f;
        particles[idx].vel.w = 0.0f;
        states[idx] = localState;
    }
}

// Kernel to update particle positions based on velocities
__global__ void updatePositions(Particle *particles, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        particles[idx].pos.x += particles[idx].vel.x * deltaTime;
        particles[idx].pos.y += particles[idx].vel.y * deltaTime;
        particles[idx].pos.z += particles[idx].vel.z * deltaTime;
    }
}

// Kernel to calculate forces using a simple particle mesh method
__global__ void calculateForces(Particle *particles, float4 *forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int j = 0; j < NUM_PARTICLES; j++) {
            if (idx != j) {
                float4 r = make_float4(particles[j].pos.x - particles[idx].pos.x,
                                      particles[j].pos.y - particles[idx].pos.y,
                                      particles[j].pos.z - particles[idx].pos.z,
                                      0.0f);
                float distSquared = r.x * r.x + r.y * r.y + r.z * r.z + 1.0f; // Add softening factor
                float dist = sqrtf(distSquared);
                float invDistCube = 1.0f / (distSquared * dist);
                force.x += r.x * invDistCube;
                force.y += r.y * invDistCube;
                force.z += r.z * invDistCube;
            }
        }
        forces[idx] = force;
    }
}

// Kernel to update velocities based on forces
__global__ void updateVelocities(Particle *particles, float4 *forces, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        particles[idx].vel.x += forces[idx].x * deltaTime;
        particles[idx].vel.y += forces[idx].y * deltaTime;
        particles[idx].vel.z += forces[idx].z * deltaTime;
    }
}

// Main function to demonstrate the usage of the kernels
int main() {
    Particle *d_particles;
    float4 *d_forces;
    hiprandState *d_states;

    // Allocate memory on the device
    hipMalloc((void**)&d_particles, NUM_PARTICLES * sizeof(Particle));
    hipMalloc((void**)&d_forces, NUM_PARTICLES * sizeof(float4));
    hipMalloc((void**)&d_states, NUM_PARTICLES * sizeof(hiprandState));

    // Initialize random states on the device
    unsigned long seed = 1234;
    hiprandState *d_states_init;
    hipMalloc((void**)&d_states_init, NUM_PARTICLES * sizeof(hiprandState));
    initParticles<<<GRID_SIZE, BLOCK_SIZE>>>(d_particles, d_states_init);
    hipMemcpy(d_states, d_states_init, NUM_PARTICLES * sizeof(hiprandState), hipMemcpyDeviceToDevice);
    hipFree(d_states_init);

    // Simulation parameters
    float deltaTime = 0.01f;
    int numSteps = 10;

    // Perform the simulation
    for (int step = 0; step < numSteps; step++) {
        calculateForces<<<GRID_SIZE, BLOCK_SIZE>>>(d_particles, d_forces);
        updateVelocities<<<GRID_SIZE, BLOCK_SIZE>>>(d_particles, d_forces, deltaTime);
        updatePositions<<<GRID_SIZE, BLOCK_SIZE>>>(d_particles, deltaTime);

        // Check for errors
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Free device memory
    hipFree(d_particles);
    hipFree(d_forces);
    hipFree(d_states);

    return 0;
}