#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 4096
#define BLOCK_SIZE 256

// Structure to represent a particle
struct Particle {
    float4 pos;  // x, y, z, w (mass)
    float4 vel;  // vx, vy, vz, padding
};

// Kernel to initialize particles with random positions and velocities
__global__ void initParticles(Particle *particles, hiprandState *states, float domainSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        hiprandState localState = states[idx];
        float3 pos = {hiprand_uniform(&localState) * domainSize,
                      hiprand_uniform(&localState) * domainSize,
                      hiprand_uniform(&localState) * domainSize};
        float3 vel = {(hiprand_uniform(&localState) - 0.5f) * 2.0f,
                      (hiprand_uniform(&localState) - 0.5f) * 2.0f,
                      (hiprand_uniform(&localState) - 0.5f) * 2.0f};
        particles[idx].pos = make_float4(pos.x, pos.y, pos.z, 1.0f); // Mass set to 1 for simplicity
        particles[idx].vel = make_float4(vel.x, vel.y, vel.z, 0.0f);
        states[idx] = localState;
    }
}

// Kernel to perform a simple N-body simulation step
__global__ void nbodyStep(Particle *particles, float dt, float domainSize) {
    extern __shared__ Particle sharedParticles[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load particle data into shared memory
    if (idx < N) {
        sharedParticles[tid] = particles[idx];
    }
    __syncthreads();

    // Compute forces and update positions and velocities
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < BLOCK_SIZE && i + blockIdx.x * blockDim.x < N; ++i) {
        float4 p1 = sharedParticles[tid].pos;
        float4 p2 = sharedParticles[i].pos;
        float3 r = make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
        float rsq = r.x * r.x + r.y * r.y + r.z * r.z + 1e-6f; // Softening
        float invR3 = rsq * rsq * rsqrtf(rsq);
        float s = p2.w * invR3; // Mass of p2 * invR3
        force.x += r.x * s;
        force.y += r.y * s;
        force.z += r.z * s;
    }
    __syncthreads();

    // Update particle velocities and positions
    if (idx < N) {
        float4 v = particles[idx].vel;
        v.x += dt * force.x;
        v.y += dt * force.y;
        v.z += dt * force.z;
        particles[idx].vel = v;

        float4 p = particles[idx].pos;
        p.x += dt * v.x;
        p.y += dt * v.y;
        p.z += dt * v.z;
        // Simple boundary conditions (wrap-around)
        p.x = fmodf(p.x, domainSize);
        p.y = fmodf(p.y, domainSize);
        p.z = fmodf(p.z, domainSize);
        particles[idx].pos = p;
    }
}

// Error checking macro
#define cudaCheckError(err) (cudaCheckError_(err, __FILE__, __LINE__))

// Error checking function
inline void cudaCheckError_(hipError_t err, const char *file, int line) {
    if (err != hipSuccess) {
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s\n", file, line, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    Particle *d_particles;
    hiprandState *d_states;

    // Allocate memory on the device
    cudaCheckError(hipMalloc(&d_particles, N * sizeof(Particle)));
    cudaCheckError(hipMalloc(&d_states, N * sizeof(hiprandState)));

    // Initialize random states
    unsigned int seed = 1234;
    hiprandState *d_seed;
    cudaCheckError(hipMalloc(&d_seed, sizeof(unsigned int)));
    cudaCheckError(hipMemcpy(d_seed, &seed, sizeof(unsigned int), hipMemcpyHostToDevice));
    initParticles<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles, d_states, 100.0f);
    cudaCheckError(hipGetLastError());
    cudaCheckError(hipDeviceSynchronize());

    // Perform a single step of the N-body simulation
    nbodyStep<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(Particle)>>>(d_particles, 0.01f, 100.0f);
    cudaCheckError(hipGetLastError());
    cudaCheckError(hipDeviceSynchronize());

    // Free device memory
    cudaCheckError(hipFree(d_particles));
    cudaCheckError(hipFree(d_states));
    cudaCheckError(hipFree(d_seed));

    return 0;
}