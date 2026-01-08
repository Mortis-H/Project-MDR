#include <iostream>
#include <hip/hip_runtime.h>


#define BLOCK_SIZE 256
#define NUM_PARTICLES 1024
#define DIM 128

__device__ float grid[DIM][DIM][DIM];

// Kernel to update particle positions and interpolate to grid
__global__ void updateParticlesAndInterpolate(float4* particles, float4* forces, int numParticles) {
    extern __shared__ float4 sharedParticles[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        sharedParticles[threadIdx.x] = particles[idx];
    }
    __syncthreads();

    if (idx < numParticles) {
        float4 p = sharedParticles[threadIdx.x];
        float4 f = forces[idx];

        // Update particle position
        p.x += f.x;
        p.y += f.y;
        p.z += f.z;

        // Interpolate particle to grid
        int ix = static_cast<int>(p.x);
        int iy = static_cast<int>(p.y);
        int iz = static_cast<int>(p.z);

        atomicAdd(&grid[ix][iy][iz], p.w);

        // Write updated particle back
        particles[idx] = p;
    }
}

// Kernel to compute forces on particles using grid data
__global__ void computeForces(float4* particles, float4* forces, int numParticles) {
    extern __shared__ float4 sharedParticles[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        sharedParticles[threadIdx.x] = particles[idx];
    }
    __syncthreads();

    if (idx < numParticles) {
        float4 p = sharedParticles[threadIdx.x];
        float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Compute force based on grid data
        int ix = static_cast<int>(p.x);
        int iy = static_cast<int>(p.y);
        int iz = static_cast<int>(p.z);

        float cellDensity = grid[ix][iy][iz];
        force.x = cellDensity;
        force.y = cellDensity;
        force.z = cellDensity;

        // Write computed force
        forces[idx] = force;
    }
}

int main() {
    float4* d_particles;
    float4* d_forces;
    float4* h_particles = new float4[NUM_PARTICLES];
    float4* h_forces = new float4[NUM_PARTICLES];

    // Initialize particles
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_particles[i] = make_float4(static_cast<float>(i % DIM), static_cast<float>((i / DIM) % DIM), static_cast<float>(i / (DIM * DIM)), 1.0f);
        h_forces[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Allocate device memory
    hipMalloc((void**)&d_particles, NUM_PARTICLES * sizeof(float4));
    hipMalloc((void**)&d_forces, NUM_PARTICLES * sizeof(float4));

    // Copy data to device
    hipMemcpy(d_particles, h_particles, NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_forces, h_forces, NUM_PARTICLES * sizeof(float4), hipMemcpyHostToDevice);

    // Launch kernels
    dim3 blocks((NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);

    updateParticlesAndInterpolate<<<blocks, threads, BLOCK_SIZE * sizeof(float4)>>>(d_particles, d_forces, NUM_PARTICLES);
    hipDeviceSynchronize();
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel updateParticlesAndInterpolate failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    computeForces<<<blocks, threads, BLOCK_SIZE * sizeof(float4)>>>(d_particles, d_forces, NUM_PARTICLES);
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel computeForces failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    hipMemcpy(h_particles, d_particles, NUM_PARTICLES * sizeof(float4), hipMemcpyDeviceToHost);
    hipMemcpy(h_forces, d_forces, NUM_PARTICLES * sizeof(float4), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_particles);
    hipFree(d_forces);

    // Clean up
    delete[] h_particles;
    delete[] h_forces;

    std::cout << "Simulation completed successfully." << std::endl;
    return 0;
}