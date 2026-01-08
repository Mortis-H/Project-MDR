#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

// Constants
const int NUM_PARTICLES = 16384;
const int BLOCK_SIZE = 256;
const float DT = 0.001f;
const float EPSILON = 1e-5f;

// Structure to represent a particle
struct Particle {
    float4 pos; // x, y, z, w (w is unused)
    float4 vel; // vx, vy, vz, vw (vw is unused)
    float4 acc; // ax, ay, az, aw (aw is unused)
};

// Function to calculate the squared distance between two particles
__device__ float4 calculateSquaredDistance(const float4& p1, const float4& p2) {
    float4 diff;
    diff.x = p1.x - p2.x;
    diff.y = p1.y - p2.y;
    diff.z = p1.z - p2.z;
    diff.w = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    return diff;
}

// Kernel to compute forces (accelerations) on each particle
__global__ void computeForces(Particle* particles, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float4 force = {0.0f, 0.0f, 0.0f, 0.0f};

    // Shared memory for particle positions
    __shared__ float4 sharedPos[BLOCK_SIZE];

    // Load particle positions into shared memory
    sharedPos[threadIdx.x] = particles[idx].pos;
    __syncthreads();

    // Calculate forces using shared memory
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        float4 diff = calculateSquaredDistance(sharedPos[i], particles[idx].pos);
        if (diff.w > EPSILON) {
            float rinv = rsqrtf(diff.w);
            float r6inv = rinv * rinv * rinv * rinv * rinv * rinv;
            float f = 48.0f * r6inv * (r6inv - 0.5f) * rinv;
            force.x += f * diff.x;
            force.y += f * diff.y;
            force.z += f * diff.z;
        }
    }
    __syncthreads();

    // Write the calculated force back to global memory
    particles[idx].acc.x = force.x;
    particles[idx].acc.y = force.y;
    particles[idx].acc.z = force.z;
}

// Kernel to update positions and velocities using velocity Verlet algorithm
__global__ void updateParticles(Particle* particles, int numParticles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float dt2 = dt * dt;
    float half_dt = 0.5f * dt;

    // Update position
    particles[idx].pos.x += particles[idx].vel.x * dt + 0.5f * particles[idx].acc.x * dt2;
    particles[idx].pos.y += particles[idx].vel.y * dt + 0.5f * particles[idx].acc.y * dt2;
    particles[idx].pos.z += particles[idx].vel.z * dt + 0.5f * particles[idx].acc.z * dt2;

    // Update velocity
    particles[idx].vel.x += half_dt * particles[idx].acc.x;
    particles[idx].vel.y += half_dt * particles[idx].acc.y;
    particles[idx].vel.z += half_dt * particles[idx].acc.z;
}

int main() {
    // Allocate and initialize host memory
    Particle* h_particles = new Particle[NUM_PARTICLES];
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_particles[i].pos = {static_cast<float>(i) * 0.1f, 0.0f, 0.0f, 0.0f};
        h_particles[i].vel = {0.0f, 0.0f, 0.0f, 0.0f};
        h_particles[i].acc = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    // Allocate device memory
    Particle* d_particles;
    hipError_t err = hipMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory" << std::endl;
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_particles, h_particles, NUM_PARTICLES * sizeof(Particle), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from host to device" << std::endl;
        hipFree(d_particles);
        delete[] h_particles;
        return -1;
    }

    // Define grid and block dimensions
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((NUM_PARTICLES + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Run the kernels
    computeForces<<<blocksPerGrid, threadsPerBlock>>>(d_particles, NUM_PARTICLES);
    hipDeviceSynchronize();
    updateParticles<<<blocksPerGrid, threadsPerBlock>>>(d_particles, NUM_PARTICLES, DT);
    hipDeviceSynchronize();

    // Copy results back to host
    err = hipMemcpy(h_particles, d_particles, NUM_PARTICLES * sizeof(Particle), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device to host" << std::endl;
        hipFree(d_particles);
        delete[] h_particles;
        return -1;
    }

    // Free device memory
    hipFree(d_particles);

    // Clean up host memory
    delete[] h_particles;

    return 0;
}