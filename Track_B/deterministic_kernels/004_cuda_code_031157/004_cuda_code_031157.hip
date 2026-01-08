#include <iostream>
#include <hip/hip_runtime.h>


#define NUM_PARTICLES 32
#define DIM 3
#define BLOCK_SIZE 16
#define NUM_BLOCKS (NUM_PARTICLES / BLOCK_SIZE)

__device__ float forces[NUM_PARTICLES][DIM];
__device__ float positions[NUM_PARTICLES][DIM];
__device__ float velocities[NUM_PARTICLES][DIM];
__device__ float accelerations[NUM_PARTICLES][DIM];
__device__ float new_positions[NUM_PARTICLES][DIM];
__device__ float new_velocities[NUM_PARTICLES][DIM];
__constant__ float dt = 0.001f;
__constant__ float epsilon = 1.0f;
__constant__ float sigma = 1.0f;
__constant__ float mass = 1.0f;

__global__ void compute_forces() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    forces[idx][0] = 0.0f;
    forces[idx][1] = 0.0f;
    forces[idx][2] = 0.0f;

    for (int j = 0; j < NUM_PARTICLES; ++j) {
        if (j == idx) continue;

        float r[3];
        float r2 = 0.0f;
        for (int d = 0; d < DIM; ++d) {
            r[d] = positions[idx][d] - positions[j][d];
            r2 += r[d] * r[d];
        }

        float r6 = r2 * r2 * r2;
        float force = 24.0f * epsilon * ((2.0f * sigma * sigma * sigma * sigma * sigma * sigma / r6) / r6 - 1.0f / r6) / r2;

        for (int d = 0; d < DIM; ++d) {
            forces[idx][d] += force * r[d];
        }
    }
}

__global__ void update_positions_and_velocities() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    for (int d = 0; d < DIM; ++d) {
        new_positions[idx][d] = positions[idx][d] + velocities[idx][d] * dt + 0.5f * accelerations[idx][d] * dt * dt;
        new_velocities[idx][d] = velocities[idx][d] + 0.5f * (forces[idx][d] / mass) * dt;
    }
}

__global__ void update_accelerations() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    for (int d = 0; d < DIM; ++d) {
        accelerations[idx][d] = forces[idx][d] / mass;
    }
}

__global__ void swap_positions_and_velocities() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    for (int d = 0; d < DIM; ++d) {
        positions[idx][d] = new_positions[idx][d];
        velocities[idx][d] = new_velocities[idx][d];
    }
}

void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Initialize positions, velocities, and accelerations
    float h_positions[NUM_PARTICLES][DIM];
    float h_velocities[NUM_PARTICLES][DIM];
    float h_accelerations[NUM_PARTICLES][DIM];

    for (int i = 0; i < NUM_PARTICLES; ++i) {
        for (int d = 0; d < DIM; ++d) {
            h_positions[i][d] = static_cast<float>(i) + static_cast<float>(d) * 0.1f;
            h_velocities[i][d] = 0.0f;
            h_accelerations[i][d] = 0.0f;
        }
    }

    // Allocate device memory
    float *d_positions, *d_velocities, *d_accelerations;
    size_t size = NUM_PARTICLES * DIM * sizeof(float);

    checkCudaError(hipMalloc(&d_positions, size), "hipMalloc d_positions");
    checkCudaError(hipMalloc(&d_velocities, size), "hipMalloc d_velocities");
    checkCudaError(hipMalloc(&d_accelerations, size), "hipMalloc d_accelerations");

    // Copy data to device
    checkCudaError(hipMemcpy(d_positions, h_positions, size, hipMemcpyHostToDevice), "hipMemcpy d_positions");
    checkCudaError(hipMemcpy(d_velocities, h_velocities, size, hipMemcpyHostToDevice), "hipMemcpy d_velocities");
    checkCudaError(hipMemcpy(d_accelerations, h_accelerations, size, hipMemcpyHostToDevice), "hipMemcpy d_accelerations");

    // Set device pointers
    hipMemcpyToSymbol(HIP_SYMBOL(positions), &d_positions, sizeof(float*));
    hipMemcpyToSymbol(HIP_SYMBOL(velocities), &d_velocities, sizeof(float*));
    hipMemcpyToSymbol(HIP_SYMBOL(accelerations), &d_accelerations, sizeof(float*));

    // Main simulation loop
    int num_steps = 100;
    for (int step = 0; step < num_steps; ++step) {
        compute_forces<<<NUM_BLOCKS, BLOCK_SIZE>>>();
        checkCudaError(hipGetLastError(), "compute_forces kernel launch");

        update_accelerations<<<NUM_BLOCKS, BLOCK_SIZE>>>();
        checkCudaError(hipGetLastError(), "update_accelerations kernel launch");

        update_positions_and_velocities<<<NUM_BLOCKS, BLOCK_SIZE>>>();
        checkCudaError(hipGetLastError(), "update_positions_and_velocities kernel launch");

        swap_positions_and_velocities<<<NUM_BLOCKS, BLOCK_SIZE>>>();
        checkCudaError(hipGetLastError(), "swap_positions_and_velocities kernel launch");

        hipDeviceSynchronize();
    }

    // Copy results back to host
    checkCudaError(hipMemcpy(h_positions, d_positions, size, hipMemcpyDeviceToHost), "hipMemcpy h_positions");

    // Free device memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_accelerations);

    return 0;
}