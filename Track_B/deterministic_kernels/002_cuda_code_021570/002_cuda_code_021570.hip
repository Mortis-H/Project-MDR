#include <iostream>
#include <hip/hip_runtime.h>

#include <hiprand/hiprand_kernel.h>

#define N 64
#define DT 0.01f
#define G 0.0000000000667430f

// Structure to represent a particle
struct Particle {
    float4 pos;  // Position (x, y, z, w)
    float4 vel;  // Velocity (vx, vy, vz, w)
};

// CUDA kernel to compute forces and update positions and velocities
__global__ void nbody_kernel(Particle *p, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Compute the force on particle i
    for (int j = 0; j < n; j++) {
        if (i == j) continue;

        float4 r = make_float4(p[j].pos.x - p[i].pos.x,
                               p[j].pos.y - p[i].pos.y,
                               p[j].pos.z - p[i].pos.z,
                               0.0f);

        float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 1e-10f;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        force.x += G * p[j].pos.w * r.x * invDist3;
        force.y += G * p[j].pos.w * r.y * invDist3;
        force.z += G * p[j].pos.w * r.z * invDist3;
    }

    // Update position and velocity
    p[i].vel.x += force.x * dt / p[i].pos.w;
    p[i].vel.y += force.y * dt / p[i].pos.w;
    p[i].vel.z += force.z * dt / p[i].pos.w;

    p[i].pos.x += p[i].vel.x * dt;
    p[i].pos.y += p[i].vel.y * dt;
    p[i].pos.z += p[i].vel.z * dt;
}

// Function to initialize particles with random positions and velocities
__global__ void init_particles(Particle *p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    hiprandState state;
    hiprand_init(i, 0, 0, &state);

    p[i].pos.x = hiprand_uniform(&state) * 100.0f - 50.0f;
    p[i].pos.y = hiprand_uniform(&state) * 100.0f - 50.0f;
    p[i].pos.z = hiprand_uniform(&state) * 100.0f - 50.0f;
    p[i].pos.w = 1.0f + hiprand_uniform(&state) * 10.0f; // Mass

    p[i].vel.x = hiprand_uniform(&state) * 2.0f - 1.0f;
    p[i].vel.y = hiprand_uniform(&state) * 2.0f - 1.0f;
    p[i].vel.z = hiprand_uniform(&state) * 2.0f - 1.0f;
}

int main() {
    Particle *h_particles, *d_particles;
    h_particles = (Particle *)malloc(N * sizeof(Particle));

    hipError_t err = hipMalloc((void **)&d_particles, N * sizeof(Particle));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory (error code " << err << ")!\n";
        exit(EXIT_FAILURE);
    }

    // Initialize particles on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    init_particles<<<numBlocks, blockSize>>>(d_particles, N);
    hipDeviceSynchronize();

    // Perform N-body simulation
    for (int step = 0; step < 100; step++) {
        nbody_kernel<<<numBlocks, blockSize>>>(d_particles, N, DT);
        hipDeviceSynchronize();
    }

    // Copy results back to host
    hipMemcpy(h_particles, d_particles, N * sizeof(Particle), hipMemcpyDeviceToHost);

    // Clean up
    free(h_particles);
    hipFree(d_particles);

    return 0;
}