#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 8192
#define BLOCK_SIZE 256

struct Particle {
    float x, y, z; // Position
    float vx, vy, vz; // Velocity
    float mass; // Mass
};

__global__ void initializeParticles(Particle *p, hiprandState *states, int seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        hiprandState localState = states[id];
        hiprand_init(seed, id, 0, &localState);
        p[id].x = hiprand_uniform(&localState) * 100.0f;
        p[id].y = hiprand_uniform(&localState) * 100.0f;
        p[id].z = hiprand_uniform(&localState) * 100.0f;
        p[id].vx = hiprand_uniform(&localState) * 0.1f - 0.05f;
        p[id].vy = hiprand_uniform(&localState) * 0.1f - 0.05f;
        p[id].vz = hiprand_uniform(&localState) * 0.1f - 0.05f;
        p[id].mass = hiprand_uniform(&localState) * 1.0f + 0.1f;
        states[id] = localState;
    }
}

__global__ void nbodyKernel(Particle *p, float dt, int n) {
    extern __shared__ Particle sdata[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * blockDim.x + tid;

    // Load one particle into shared memory
    if (i < n) {
        sdata[tid] = p[i];
    }
    __syncthreads();

    // Compute forces for each particle
    if (i < n) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        for (int j = 0; j < blockDim.x; j++) {
            if (i != j) {
                float dx = sdata[j].x - p[i].x;
                float dy = sdata[j].y - p[i].y;
                float dz = sdata[j].z - p[i].z;
                float r2 = dx * dx + dy * dy + dz * dz + 1e-6f;
                float r = sqrtf(r2);
                float f = sdata[j].mass / (r * r2);
                ax += f * dx;
                ay += f * dy;
                az += f * dz;
            }
        }
        p[i].vx += ax * dt;
        p[i].vy += ay * dt;
        p[i].vz += az * dt;
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

int main() {
    Particle *h_particles, *d_particles;
    hiprandState *d_states;

    // Allocate host memory
    h_particles = (Particle *)malloc(N * sizeof(Particle));

    // Allocate device memory
    hipError_t err = hipMalloc((void **)&d_particles, N * sizeof(Particle));
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_states, N * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize particles on GPU
    int seed = 1234;
    initializeParticles<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles, d_states, seed);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Perform N-body simulation
    float dt = 0.01f;
    int steps = 100;
    for (int step = 0; step < steps; step++) {
        nbodyKernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(Particle)>>>(d_particles, dt, N);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy results back to host
    err = hipMemcpy(h_particles, d_particles, N * sizeof(Particle), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_particles);
    hipFree(d_states);

    // Free host memory
    free(h_particles);

    std::cout << "N-body simulation completed successfully." << std::endl;

    return 0;
}