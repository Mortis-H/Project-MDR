#include <iostream>
#include <hip/hip_runtime.h>

#include <hiprand/hiprand_kernel.h>
#include <cmath>

#define N 256
#define BLOCK_SIZE 128

__constant__ float4 d_particles[N];
__constant__ float4 d_forces[N];

__device__ __inline__ float4 calculateForce(float4 particle, const float4* particles, int numParticles) {
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j = 0; j < numParticles; ++j) {
        float4 other = particles[j];
        float dx = other.x - particle.x;
        float dy = other.y - particle.y;
        float dz = other.z - particle.z;
        float r2 = dx * dx + dy * dy + dz * dz;
        float r = sqrtf(r2);
        float f = 1.0f / (r2 * r); // Simple force calculation
        force.x += dx * f;
        force.y += dy * f;
        force.z += dz * f;
    }
    return force;
}

__global__ void nbodyKernel(int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        float4 particle = d_particles[idx];
        float4 force = calculateForce(particle, d_particles, numParticles);
        d_forces[idx] = force;
    }
}

__global__ void initializeParticles(float4* particles, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        hiprandState state;
        hiprand_init((unsigned long long) clock() + idx, 0, 0, &state);
        particles[idx] = make_float4(hiprand_uniform(&state) * 100.0f,
                                     hiprand_uniform(&state) * 100.0f,
                                     hiprand_uniform(&state) * 100.0f,
                                     0.0f);
    }
}

int main() {
    float4* h_particles = new float4[N];
    float4* h_forces = new float4[N];
    float4* d_particles;
    float4* d_forces;

    hipError_t err;

    // Allocate memory on device
    err = hipMalloc(&d_particles, N * sizeof(float4));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for particles: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc(&d_forces, N * sizeof(float4));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for forces: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize particles on device
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializeParticles<<<blocks, BLOCK_SIZE>>>(d_particles, N);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching initializeParticles kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy particles to constant memory
    err = hipMemcpyToSymbol(HIP_SYMBOL(::d_particles), d_particles, N * sizeof(float4));
    if (err != hipSuccess) {
        std::cerr << "Error copying particles to constant memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch N-body kernel
    nbodyKernel<<<blocks, BLOCK_SIZE>>>(N);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching nbodyKernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy forces back to host
    err = hipMemcpyFromSymbol(h_forces, HIP_SYMBOL(::d_forces), N * sizeof(float4));
    if (err != hipSuccess) {
        std::cerr << "Error copying forces from constant memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_particles);
    hipFree(d_forces);

    // Clean up host memory
    delete[] h_particles;
    delete[] h_forces;

    return 0;
}