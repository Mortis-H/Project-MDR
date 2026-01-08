#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

#define N 256
#define DT 0.001f
#define EPSILON 1e-10f

__global__ void leapfrog_kernel(float4* pos, float4* vel, float4* force, float mass) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load position and velocity
    float4 p = pos[idx];
    float4 v = vel[idx];
    float4 f = force[idx];

    // First half kick (update velocity)
    v.x += (f.x / mass) * (DT / 2.0f);
    v.y += (f.y / mass) * (DT / 2.0f);
    v.z += (f.z / mass) * (DT / 2.0f);

    // Drift (update position)
    p.x += v.x * DT;
    p.y += v.y * DT;
    p.z += v.z * DT;

    // Apply periodic boundary conditions
    p.x = fmodf(p.x + 1.0f, 1.0f);
    p.y = fmodf(p.y + 1.0f, 1.0f);
    p.z = fmodf(p.z + 1.0f, 1.0f);

    // Second half kick (update velocity)
    v.x += (f.x / mass) * (DT / 2.0f);
    v.y += (f.y / mass) * (DT / 2.0f);
    v.z += (f.z / mass) * (DT / 2.0f);

    // Store updated position and velocity
    pos[idx] = p;
    vel[idx] = v;
}

__global__ void compute_forces_kernel(float4* pos, float4* force, float epsilon) {
    extern __shared__ float4 shared_pos[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int tid = threadIdx.x;
    shared_pos[tid] = pos[idx];
    __syncthreads();

    float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Compute forces using Lennard-Jones potential
    for (int j = 0; j < blockDim.x; ++j) {
        if (tid != j) {
            float4 r = make_float4(shared_pos[j].x - shared_pos[tid].x,
                                   shared_pos[j].y - shared_pos[tid].y,
                                   shared_pos[j].z - shared_pos[tid].z, 0.0f);

            // Apply periodic boundary conditions
            r.x -= roundf(r.x);
            r.y -= roundf(r.y);
            r.z -= roundf(r.z);

            float rsq = r.x * r.x + r.y * r.y + r.z * r.z + EPSILON;
            float r2inv = 1.0f / rsq;
            float r6inv = r2inv * r2inv * r2inv;
            float force_magnitude = 24.0f * r2inv * r6inv * (2.0f * r6inv - 1.0f);

            f.x += force_magnitude * r.x;
            f.y += force_magnitude * r.y;
            f.z += force_magnitude * r.z;
        }
    }

    __syncthreads();

    // Write out the computed force
    force[idx] = f;
}

int main() {
    float4* d_pos, *d_vel, *d_force;
    float4* h_pos = new float4[N];
    float4* h_vel = new float4[N];
    float4* h_force = new float4[N];

    // Initialize positions and velocities
    for (int i = 0; i < N; ++i) {
        h_pos[i] = make_float4(static_cast<float>(rand()) / RAND_MAX,
                               static_cast<float>(rand()) / RAND_MAX,
                               static_cast<float>(rand()) / RAND_MAX, 0.0f);
        h_vel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_pos, N * sizeof(float4));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for positions" << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_vel, N * sizeof(float4));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for velocities" << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_force, N * sizeof(float4));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for forces" << std::endl;
        return -1;
    }

    // Copy data to device
    err = hipMemcpy(d_pos, h_pos, N * sizeof(float4), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy positions to device" << std::endl;
        return -1;
    }
    err = hipMemcpy(d_vel, h_vel, N * sizeof(float4), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy velocities to device" << std::endl;
        return -1;
    }

    // Define block and grid sizes
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // Simulate for a number of steps
    int steps = 100;
    float mass = 1.0f;

    for (int step = 0; step < steps; ++step) {
        // Compute forces
        compute_forces_kernel<<<gridSize, blockSize, blockSize.x * sizeof(float4)>>>(d_pos, d_force, EPSILON);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in compute_forces_kernel: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Leapfrog integration
        leapfrog_kernel<<<gridSize, blockSize>>>(d_pos, d_vel, d_force, mass);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in leapfrog_kernel: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy results back to host
    err = hipMemcpy(h_pos, d_pos, N * sizeof(float4), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy positions back to host" << std::endl;
        return -1;
    }
    err = hipMemcpy(h_vel, d_vel, N * sizeof(float4), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy velocities back to host" << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_pos);
    hipFree(d_vel);
    hipFree(d_force);

    // Free host memory
    delete[] h_pos;
    delete[] h_vel;
    delete[] h_force;

    return 0;
}