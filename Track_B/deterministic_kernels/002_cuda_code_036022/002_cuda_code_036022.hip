#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define N 1024
#define BLOCK_SIZE 256

__constant__ float c_dt = 0.001f;
__constant__ float c_cutoff = 1.0f;
__constant__ float c_mass = 1.0f;

struct Particle {
    float4 pos; // x, y, z, w (w is unused)
    float4 vel; // vx, vy, vz, w (w is unused)
    float4 acc; // ax, ay, az, w (w is unused)
};

__global__ void computeForces(Particle *d_particles, float4 *d_forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float4 pos = d_particles[idx].pos;
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int j = 0; j < N; j++) {
        if (j == idx) continue;

        float4 pos_j = d_particles[j].pos;
        float4 r = make_float4(pos.x - pos_j.x, pos.y - pos_j.y, pos.z - pos_j.z, 0.0f);
        float rsq = r.x * r.x + r.y * r.y + r.z * r.z;

        if (rsq < c_cutoff * c_cutoff) {
            float rinv = rsq > 0.0f ? rsqrtf(rsq) : 0.0f;
            float r6inv = rinv * rinv * rinv * rinv * rinv * rinv;
            float force_mag = 24.0f * (2.0f * r6inv * r6inv - r6inv) * rinv * rinv;
            force.x += force_mag * r.x;
            force.y += force_mag * r.y;
            force.z += force_mag * r.z;
        }
    }

    d_forces[idx] = force;
}

__global__ void updateParticles(Particle *d_particles, float4 *d_forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float4 pos = d_particles[idx].pos;
    float4 vel = d_particles[idx].vel;
    float4 acc = d_particles[idx].acc;
    float4 force = d_forces[idx];

    float4 new_acc = make_float4(force.x / c_mass, force.y / c_mass, force.z / c_mass, 0.0f);
    float4 new_vel = make_float4(vel.x + 0.5f * c_dt * (acc.x + new_acc.x),
                                vel.y + 0.5f * c_dt * (acc.y + new_acc.y),
                                vel.z + 0.5f * c_dt * (acc.z + new_acc.z),
                                0.0f);
    float4 new_pos = make_float4(pos.x + c_dt * new_vel.x,
                                pos.y + c_dt * new_vel.y,
                                pos.z + c_dt * new_vel.z,
                                0.0f);

    d_particles[idx].pos = new_pos;
    d_particles[idx].vel = new_vel;
    d_particles[idx].acc = new_acc;
}

int main() {
    hipError_t err;

    // Allocate host memory
    std::vector<Particle> h_particles(N);
    std::vector<float4> h_forces(N);

    // Initialize particles
    for (int i = 0; i < N; i++) {
        h_particles[i].pos = make_float4(static_cast<float>(i % 32) * 0.1f,
                                         static_cast<float>(i / 32 % 32) * 0.1f,
                                         static_cast<float>(i / 1024) * 0.1f,
                                         0.0f);
        h_particles[i].vel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        h_particles[i].acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Allocate device memory
    Particle *d_particles;
    float4 *d_forces;
    err = hipMalloc((void**)&d_particles, N * sizeof(Particle));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_particles: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_forces, N * sizeof(float4));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_forces: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy initial particle data to device
    err = hipMemcpy(d_particles, h_particles.data(), N * sizeof(Particle), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying h_particles to d_particles: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Set up asynchronous memory copy
    hipStream_t stream;
    err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
        std::cerr << "Error creating stream: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Perform molecular dynamics simulation
    for (int step = 0; step < 100; step++) {
        // Compute forces
        computeForces<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(d_particles, d_forces);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in computeForces kernel: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Update particles
        updateParticles<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(d_particles, d_forces);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in updateParticles kernel: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Asynchronously copy updated particles back to host
        err = hipMemcpyAsync(h_particles.data(), d_particles, N * sizeof(Particle), hipMemcpyDeviceToHost, stream);
        if (err != hipSuccess) {
            std::cerr << "Error copying d_particles to h_particles: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Synchronize to ensure all operations are complete before the next iteration
        err = hipStreamSynchronize(stream);
        if (err != hipSuccess) {
            std::cerr << "Error synchronizing stream: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Clean up
    err = hipFree(d_particles);
    if (err != hipSuccess) {
        std::cerr << "Error freeing d_particles: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipFree(d_forces);
    if (err != hipSuccess) {
        std::cerr << "Error freeing d_forces: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipStreamDestroy(stream);
    if (err != hipSuccess) {
        std::cerr << "Error destroying stream: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    return 0;
}