#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Constants for the simulation
#define NUM_PARTICLES 64
#define DT 0.001f
#define CUTOFF 3.0f
#define EPSILON 1.0f
#define SIGMA 1.0f

// Kernel to initialize random positions and velocities
__global__ void initParticles(float4 *pos, float4 *vel, hiprandState *states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        hiprandState localState = states[idx];
        hiprand_init(seed, idx, 0, &localState);
        pos[idx] = make_float4(hiprand_uniform(&localState) * 10.0f,
                               hiprand_uniform(&localState) * 10.0f,
                               hiprand_uniform(&localState) * 10.0f, 0.0f);
        vel[idx] = make_float4(hiprand_normal(&localState) * 0.1f,
                               hiprand_normal(&localState) * 0.1f,
                               hiprand_normal(&localState) * 0.1f, 0.0f);
        states[idx] = localState;
    }
}

// Kernel to compute forces using Lennard-Jones potential
__global__ void computeForces(float4 *pos, float4 *f, float *mass) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        f[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int j = 0; j < NUM_PARTICLES; j++) {
            if (j != idx) {
                float3 r = make_float3(pos[j].x - pos[idx].x,
                                       pos[j].y - pos[idx].y,
                                       pos[j].z - pos[idx].z);
                float rsq = r.x * r.x + r.y * r.y + r.z * r.z;
                if (rsq < CUTOFF * CUTOFF) {
                    float r2inv = 1.0f / rsq;
                    float r6inv = r2inv * r2inv * r2inv;
                    float force = 24.0f * EPSILON * (2.0f * r6inv * r6inv - r6inv) * r2inv;
                    f[idx].x += force * r.x;
                    f[idx].y += force * r.y;
                    f[idx].z += force * r.z;
                }
            }
        }
    }
}

// Kernel to update positions and velocities using velocity Verlet
__global__ void updatePositionsAndVelocities(float4 *pos, float4 *vel, float4 *f, float *mass, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        float3 a = make_float3(f[idx].x / mass[idx], f[idx].y / mass[idx], f[idx].z / mass[idx]);
        pos[idx].x += vel[idx].x * dt + 0.5f * a.x * dt * dt;
        pos[idx].y += vel[idx].y * dt + 0.5f * a.y * dt * dt;
        pos[idx].z += vel[idx].z * dt + 0.5f * a.z * dt * dt;
        vel[idx].x += 0.5f * a.x * dt;
        vel[idx].y += 0.5f * a.y * dt;
        vel[idx].z += 0.5f * a.z * dt;
    }
}

int main() {
    // Allocate host memory
    float4 *h_pos = (float4 *)malloc(NUM_PARTICLES * sizeof(float4));
    float4 *h_vel = (float4 *)malloc(NUM_PARTICLES * sizeof(float4));
    float4 *h_f = (float4 *)malloc(NUM_PARTICLES * sizeof(float4));
    float *h_mass = (float *)malloc(NUM_PARTICLES * sizeof(float));

    // Allocate device memory
    float4 *d_pos, *d_vel, *d_f;
    float *d_mass;
    hiprandState *d_states;
    hipMalloc((void **)&d_pos, NUM_PARTICLES * sizeof(float4));
    hipMalloc((void **)&d_vel, NUM_PARTICLES * sizeof(float4));
    hipMalloc((void **)&d_f, NUM_PARTICLES * sizeof(float4));
    hipMalloc((void **)&d_mass, NUM_PARTICLES * sizeof(float));
    hipMalloc((void **)&d_states, NUM_PARTICLES * sizeof(hiprandState));

    // Initialize masses
    for (int i = 0; i < NUM_PARTICLES; i++) {
        h_mass[i] = 1.0f;
    }
    hipMemcpy(d_mass, h_mass, NUM_PARTICLES * sizeof(float), hipMemcpyHostToDevice);

    // Initialize particles
    initParticles<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_pos, d_vel, d_states, time(NULL));
    hipDeviceSynchronize();

    // Allocate pinned memory for asynchronous data transfer
    float4 *h_pos_pinned, *h_vel_pinned, *h_f_pinned;
    hipHostAlloc((void **)&h_pos_pinned, NUM_PARTICLES * sizeof(float4), hipHostMallocDefault);
    hipHostAlloc((void **)&h_vel_pinned, NUM_PARTICLES * sizeof(float4), hipHostMallocDefault);
    hipHostAlloc((void **)&h_f_pinned, NUM_PARTICLES * sizeof(float4), hipHostMallocDefault);

    // Simulation loop
    int numSteps = 1000;
    for (int step = 0; step < numSteps; step++) {
        // Asynchronously copy positions to host
        hipMemcpyAsync(h_pos_pinned, d_pos, NUM_PARTICLES * sizeof(float4), hipMemcpyDeviceToHost, 0);

        // Compute forces
        computeForces<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_pos, d_f, d_mass);
        hipDeviceSynchronize();

        // Asynchronously copy forces to host
        hipMemcpyAsync(h_f_pinned, d_f, NUM_PARTICLES * sizeof(float4), hipMemcpyDeviceToHost, 0);

        // Update positions and velocities
        updatePositionsAndVelocities<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_pos, d_vel, d_f, d_mass, DT);
        hipDeviceSynchronize();

        // Asynchronously copy velocities to host
        hipMemcpyAsync(h_vel_pinned, d_vel, NUM_PARTICLES * sizeof(float4), hipMemcpyDeviceToHost, 0);
    }

    // Free memory
    hipFree(d_pos);
    hipFree(d_vel);
    hipFree(d_f);
    hipFree(d_mass);
    hipFree(d_states);
    hipHostFree(h_pos_pinned);
    hipHostFree(h_vel_pinned);
    hipHostFree(h_f_pinned);
    free(h_pos);
    free(h_vel);
    free(h_f);
    free(h_mass);

    return 0;
}