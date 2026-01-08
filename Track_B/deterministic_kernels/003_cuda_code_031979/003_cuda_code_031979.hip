#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARTICLES 256
#define BLOCK_SIZE 128
#define DIM 3
#define EPSILON 1e-6

__constant__ float k = 1.0f;
__constant__ float dt = 0.001f;

struct Particle {
    float r[DIM];
    float v[DIM];
    float a[DIM];
};

__global__ void initParticles(Particle *particles, hiprandState *states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        hiprandState localState = states[idx];
        hiprand_init(seed, idx, 0, &localState);
        for (int d = 0; d < DIM; ++d) {
            particles[idx].r[d] = hiprand_uniform(&localState) * 10.0f;
            particles[idx].v[d] = hiprand_uniform(&localState) * 2.0f - 1.0f;
            particles[idx].a[d] = 0.0f;
        }
        states[idx] = localState;
    }
}

__device__ void computeForce(Particle &p, Particle *particles) {
    float force[DIM] = {0.0f, 0.0f, 0.0f};
    for (int j = 0; j < NUM_PARTICLES; ++j) {
        float r[DIM], rij[DIM];
        float rsq = 0.0f;
        for (int d = 0; d < DIM; ++d) {
            rij[d] = particles[j].r[d] - p.r[d];
            rsq += rij[d] * rij[d];
        }
        if (rsq > EPSILON) {
            float r2inv = 1.0f / rsq;
            float r6inv = r2inv * r2inv * r2inv;
            float f = 24.0f * k * (2.0f * r6inv * r6inv - r6inv) * r2inv;
            for (int d = 0; d < DIM; ++d) {
                force[d] += f * rij[d];
            }
        }
    }
    for (int d = 0; d < DIM; ++d) {
        p.a[d] = force[d];
    }
}

__global__ void velocityVerlet(Particle *particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        Particle &p = particles[idx];
        // First half kick
        for (int d = 0; d < DIM; ++d) {
            p.v[d] += 0.5f * dt * p.a[d];
        }
        // Drift
        for (int d = 0; d < DIM; ++d) {
            p.r[d] += dt * p.v[d];
        }
        // Compute new acceleration
        computeForce(p, particles);
        // Second half kick
        for (int d = 0; d < DIM; ++d) {
            p.v[d] += 0.5f * dt * p.a[d];
        }
    }
}

__global__ void resetStates(hiprandState *states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        hiprandState localState = states[idx];
        hiprand_init(seed, idx, 0, &localState);
        states[idx] = localState;
    }
}

int main() {
    Particle *d_particles;
    hiprandState *d_states;

    hipMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle));
    hipMalloc(&d_states, NUM_PARTICLES * sizeof(hiprandState));

    resetStates<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, 1234);
    hipDeviceSynchronize();

    initParticles<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles, d_states, 1234);
    hipDeviceSynchronize();

    for (int step = 0; step < 100; ++step) {
        velocityVerlet<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles);
        hipDeviceSynchronize();
    }

    hipFree(d_particles);
    hipFree(d_states);

    return 0;
}