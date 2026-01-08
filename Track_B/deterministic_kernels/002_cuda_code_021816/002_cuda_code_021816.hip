// nbody_simulation.cu
// A CUDA-based N-body simulation using particle mesh with atomic operations
// Optimized for NVIDIA Tesla architecture (Hopper)

#include <iostream>
#include <hip/hip_runtime.h>

#include <hip/hip_math_constants.h>

#define NUM_PARTICLES 256
#define GRID_SIZE 16
#define BLOCK_SIZE 16

// Particle structure
typedef struct {
    float4 pos; // x, y, z, mass
    float4 vel; // vx, vy, vz, unused
} Particle;

// Kernel to compute forces using particle mesh
__global__ void computeForces(Particle *particles, float *mesh, int meshSize, float G, float dt) {
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    Particle p = particles[idx];
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Convert particle position to mesh index
    int x = static_cast<int>(p.pos.x * meshSize);
    int y = static_cast<int>(p.pos.y * meshSize);
    int z = static_cast<int>(p.pos.z * meshSize);

    // Ensure indices are within bounds
    x = max(0, min(x, meshSize - 1));
    y = max(0, min(y, meshSize - 1));
    z = max(0, min(z, meshSize - 1));

    // Calculate the force contribution from the mesh
    float meshDensity = mesh[x * meshSize * meshSize + y * meshSize + z];
    force.x += G * meshDensity * (x - p.pos.x);
    force.y += G * meshDensity * (y - p.pos.y);
    force.z += G * meshDensity * (z - p.pos.z);

    // Update particle velocity using atomic operations
    atomicAdd(&particles[idx].vel.x, force.x * dt / p.pos.w);
    atomicAdd(&particles[idx].vel.y, force.y * dt / p.pos.w);
    atomicAdd(&particles[idx].vel.z, force.z * dt / p.pos.w);
}

// Kernel to update particle positions
__global__ void updatePositions(Particle *particles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    // Update position based on velocity
    particles[idx].pos.x += particles[idx].vel.x * dt;
    particles[idx].pos.y += particles[idx].vel.y * dt;
    particles[idx].pos.z += particles[idx].vel.z * dt;
}

int main() {
    const int meshSize = 32; // Size of the particle mesh
    const float G = 6.67430e-11f; // Gravitational constant
    const float dt = 0.01f; // Time step

    // Allocate memory for particles and mesh
    Particle *h_particles = new Particle[NUM_PARTICLES];
    float *h_mesh = new float[meshSize * meshSize * meshSize];

    // Initialize particles and mesh (simple initialization for demonstration)
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_particles[i].pos = make_float4(static_cast<float>(i) / NUM_PARTICLES, 0.5f, 0.5f, 1.0f);
        h_particles[i].vel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (int i = 0; i < meshSize * meshSize * meshSize; ++i) {
        h_mesh[i] = 1.0f; // Uniform density for simplicity
    }

    // Allocate device memory
    Particle *d_particles;
    float *d_mesh;
    hipError_t err = hipMalloc((void**)&d_particles, NUM_PARTICLES * sizeof(Particle));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for particles" << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_mesh, meshSize * meshSize * meshSize * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for mesh" << std::endl;
        hipFree(d_particles);
        return -1;
    }

    // Copy data to device
    err = hipMemcpy(d_particles, h_particles, NUM_PARTICLES * sizeof(Particle), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy particles to device" << std::endl;
        hipFree(d_particles);
        hipFree(d_mesh);
        return -1;
    }
    err = hipMemcpy(d_mesh, h_mesh, meshSize * meshSize * meshSize * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy mesh to device" << std::endl;
        hipFree(d_particles);
        hipFree(d_mesh);
        return -1;
    }

    // Launch kernels
    computeForces<<<GRID_SIZE, BLOCK_SIZE, 0>>>(d_particles, d_mesh, meshSize, G, dt);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_particles);
        hipFree(d_mesh);
        return -1;
    }
    hipDeviceSynchronize();

    updatePositions<<<GRID_SIZE, BLOCK_SIZE>>>(d_particles, dt);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_particles);
        hipFree(d_mesh);
        return -1;
    }
    hipDeviceSynchronize();

    // Copy results back to host
    err = hipMemcpy(h_particles, d_particles, NUM_PARTICLES * sizeof(Particle), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy particles back to host" << std::endl;
        hipFree(d_particles);
        hipFree(d_mesh);
        return -1;
    }

    // Free device memory
    hipFree(d_particles);
    hipFree(d_mesh);

    // Output first particle's new position for demonstration
    std::cout << "Particle 0 new position: (" << h_particles[0].pos.x << ", " << h_particles[0].pos.y << ", " << h_particles[0].pos.z << ")" << std::endl;

    // Clean up
    delete[] h_particles;
    delete[] h_mesh;

    return 0;
}