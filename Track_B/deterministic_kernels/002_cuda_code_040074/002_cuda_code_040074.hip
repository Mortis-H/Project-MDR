#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARTICLES 4096
#define DT 0.001f
#define EPSILON 1e-5f
#define GRAVITY 9.81f

// Kernel to compute forces using a simple gravitational force model
__global__ void computeForces(float4* pos, float4* force, float4* vel, float4* acc, float4* accPrev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 p = pos[idx];

    // Simple gravitational force model (only z-component for simplicity)
    f.z = -GRAVITY;

    // Store computed force
    force[idx] = f;
}

// Beeman's algorithm kernel
__global__ void beemanIntegration(float4* pos, float4* vel, float4* acc, float4* accPrev, float4* force) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    float4 p = pos[idx];
    float4 v = vel[idx];
    float4 a = acc[idx];
    float4 aPrev = accPrev[idx];
    float4 f = force[idx];

    // Update position using Beeman's algorithm
    p.x += v.x * DT + (1.0f / 6.0f) * (4.0f * a.x - aPrev.x) * DT * DT;
    p.y += v.y * DT + (1.0f / 6.0f) * (4.0f * a.y - aPrev.y) * DT * DT;
    p.z += v.z * DT + (1.0f / 6.0f) * (4.0f * a.z - aPrev.z) * DT * DT;

    // Update velocity using Beeman's algorithm
    v.x += (2.0f / 3.0f) * f.x * DT + (1.0f / 6.0f) * (f.x - aPrev.x) * DT;
    v.y += (2.0f / 3.0f) * f.y * DT + (1.0f / 6.0f) * (f.y - aPrev.y) * DT;
    v.z += (2.0f / 3.0f) * f.z * DT + (1.0f / 6.0f) * (f.z - aPrev.z) * DT;

    // Update acceleration
    aPrev = a;
    a = f;

    // Store updated position, velocity, and acceleration
    pos[idx] = p;
    vel[idx] = v;
    acc[idx] = a;
    accPrev[idx] = aPrev;
}

// Error checking macro
#define cudaCheckError() {                                          \
    hipError_t e=hipGetLastError();                               \
    if(e!=hipSuccess) {                                            \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

int main() {
    // Allocate memory on the host
    float4* h_pos = new float4[NUM_PARTICLES];
    float4* h_vel = new float4[NUM_PARTICLES];
    float4* h_acc = new float4[NUM_PARTICLES];
    float4* h_accPrev = new float4[NUM_PARTICLES];
    float4* h_force = new float4[NUM_PARTICLES];

    // Initialize positions, velocities, accelerations, and previous accelerations
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_pos[i] = make_float4(static_cast<float>(i), 0.0f, 0.0f, 0.0f);
        h_vel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        h_acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        h_accPrev[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Allocate memory on the device
    float4* d_pos, *d_vel, *d_acc, *d_accPrev, *d_force;
    hipMalloc((void**)&d_pos, NUM_PARTICLES * sizeof(float4));
    hipMalloc((void**)&d_vel, NUM_PARTICLES * sizeof(float4));
    hipMalloc((void**)&d_acc, NUM_PARTICLES * sizeof(float4));
    hipMalloc((void**)&d_accPrev, NUM_PARTICLES * sizeof(float4));
    hipMalloc((void**)&d_force, NUM_PARTICLES * sizeof(float4));

    // Copy data from host to device
    hipMemcpy(d_pos, h_pos, NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_vel, h_vel, NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_acc, h_acc, NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_accPrev, h_accPrev, NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);

    // Set grid and block sizes
    int blockSize = 256;
    int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;

    // Simulation loop
    int numSteps = 1000;
    for (int step = 0; step < numSteps; ++step) {
        // Compute forces
        computeForces<<<numBlocks, blockSize>>>(d_pos, d_force, d_vel, d_acc, d_accPrev);
        cudaCheckError();

        // Integrate using Beeman's algorithm
        beemanIntegration<<<numBlocks, blockSize>>>(d_pos, d_vel, d_acc, d_accPrev, d_force);
        cudaCheckError();

        // Swap accelerations for next step
        float4* temp = d_accPrev;
        d_accPrev = d_acc;
        d_acc = temp;
    }

    // Copy results back to host
    hipMemcpy(h_pos, d_pos, NUM_PARTICLES * sizeof(float4), hipMemcpyDeviceToHost);
    hipMemcpy(h_vel, d_vel, NUM_PARTICLES * sizeof(float4), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_pos);
    hipFree(d_vel);
    hipFree(d_acc);
    hipFree(d_accPrev);
    hipFree(d_force);

    // Free host memory
    delete[] h_pos;
    delete[] h_vel;
    delete[] h_acc;
    delete[] h_accPrev;
    delete[] h_force;

    return 0;
}