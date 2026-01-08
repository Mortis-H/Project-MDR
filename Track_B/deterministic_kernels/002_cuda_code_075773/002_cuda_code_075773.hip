#include <iostream>
#include <hip/hip_runtime.h>


// Constants for the simulation
#define NUM_PARTICLES 32
#define DT 0.001f
#define G 9.81f

// Kernel to perform Beeman integration
__global__ void beemanIntegration(float4* pos, float4* vel, float4* acc, float4* acc_prev, float4* forces, float mass) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        // Calculate new acceleration
        float4 a = make_float4(forces[idx].x / mass, forces[idx].y / mass, forces[idx].z / mass, 0.0f);

        // Update position using Beeman's algorithm
        pos[idx].x += vel[idx].x * DT + (4.0f / 3.0f * a.x - 1.0f / 3.0f * acc_prev[idx].x) * DT * DT;
        pos[idx].y += vel[idx].y * DT + (4.0f / 3.0f * a.y - 1.0f / 3.0f * acc_prev[idx].y) * DT * DT;
        pos[idx].z += vel[idx].z * DT + (4.0f / 3.0f * a.z - 1.0f / 3.0f * acc_prev[idx].z) * DT * DT;

        // Update velocity using Beeman's algorithm
        vel[idx].x += (3.0f / 2.0f * a.x - 1.0f / 2.0f * acc_prev[idx].x) * DT;
        vel[idx].y += (3.0f / 2.0f * a.y - 1.0f / 2.0f * acc_prev[idx].y) * DT;
        vel[idx].z += (3.0f / 2.0f * a.z - 1.0f / 2.0f * acc_prev[idx].z) * DT;

        // Update acceleration
        acc_prev[idx] = acc[idx];
        acc[idx] = a;
    }
}

// Kernel to compute gravitational forces
__global__ void computeGravitationalForces(float4* pos, float4* forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int j = 0; j < NUM_PARTICLES; ++j) {
            if (idx != j) {
                float4 r = make_float4(pos[j].x - pos[idx].x, pos[j].y - pos[idx].y, pos[j].z - pos[idx].z, 0.0f);
                float r_squared = r.x * r.x + r.y * r.y + r.z * r.z;
                float force_magnitude = G / r_squared;
                f.x += force_magnitude * r.x;
                f.y += force_magnitude * r.y;
                f.z += force_magnitude * r.z;
            }
        }
        forces[idx] = f;
    }
}

int main() {
    // Allocate memory on host
    float4* h_pos = new float4[NUM_PARTICLES];
    float4* h_vel = new float4[NUM_PARTICLES];
    float4* h_acc = new float4[NUM_PARTICLES];
    float4* h_acc_prev = new float4[NUM_PARTICLES];
    float4* h_forces = new float4[NUM_PARTICLES];

    // Initialize positions, velocities, accelerations, and previous accelerations
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_pos[i] = make_float4(static_cast<float>(i), static_cast<float>(i), static_cast<float>(i), 0.0f);
        h_vel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        h_acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        h_acc_prev[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Allocate memory on device
    float4* d_pos, *d_vel, *d_acc, *d_acc_prev, *d_forces;
    hipMalloc((void**)&d_pos, NUM_PARTICLES * sizeof(float4));
    hipMalloc((void**)&d_vel, NUM_PARTICLES * sizeof(float4));
    hipMalloc((void**)&d_acc, NUM_PARTICLES * sizeof(float4));
    hipMalloc((void**)&d_acc_prev, NUM_PARTICLES * sizeof(float4));
    hipMalloc((void**)&d_forces, NUM_PARTICLES * sizeof(float4));

    // Copy initial data from host to device
    hipMemcpy(d_pos, h_pos, NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_vel, h_vel, NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_acc, h_acc, NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_acc_prev, h_acc_prev, NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(32);
    dim3 gridSize((NUM_PARTICLES + blockSize.x - 1) / blockSize.x);

    // Perform a few time steps of the simulation
    for (int step = 0; step < 100; ++step) {
        // Compute forces asynchronously
        computeGravitationalForces<<<gridSize, blockSize>>>(d_pos, d_forces);
        hipDeviceSynchronize();

        // Update positions and velocities asynchronously
        beemanIntegration<<<gridSize, blockSize>>>(d_pos, d_vel, d_acc, d_acc_prev, d_forces, 1.0f);
        hipDeviceSynchronize();

        // Copy positions back to host for visualization or analysis
        hipMemcpyAsync(h_pos, d_pos, NUM_PARTICLES * sizeof(float4), hipMemcpyDeviceToHost);
    }

    // Free device memory
    hipFree(d_pos);
    hipFree(d_vel);
    hipFree(d_acc);
    hipFree(d_acc_prev);
    hipFree(d_forces);

    // Free host memory
    delete[] h_pos;
    delete[] h_vel;
    delete[] h_acc;
    delete[] h_acc_prev;
    delete[] h_forces;

    return 0;
}