#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define BLOCK_SIZE 16
#define DIM 128

__global__ void initialize(hiprandState *states, float *u, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < DIM) {
        hiprand_init(seed, idx, 0, &states[idx]);
        u[idx] = hiprand_uniform(&states[idx]);
    }
}

__global__ void heatStep(float *u, float *u_new, float dt, float dx) {
    __shared__ float shared_u[BLOCK_SIZE + 2];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < DIM) {
        // Load data into shared memory with halo cells
        if (tid == 0 && blockIdx.x > 0) {
            shared_u[0] = u[idx - blockDim.x];
        }
        if (tid == blockDim.x - 1 && blockIdx.x < (DIM / blockDim.x) - 1) {
            shared_u[BLOCK_SIZE + 1] = u[idx + blockDim.x];
        }
        __syncthreads();

        shared_u[tid + 1] = u[idx];
        __syncthreads();

        // Finite difference method for heat equation
        if (tid > 0 && tid < blockDim.x - 1) {
            float laplacian = (shared_u[tid - 1] - 2 * shared_u[tid + 1] + shared_u[tid + 2]) / (dx * dx);
            u_new[idx] = u[idx] + dt * laplacian;
        }
    }
}

__global__ void update(float *u, float *u_new) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < DIM) {
        atomicExch(&u[idx], u_new[idx]);
    }
}

int main() {
    const int size = DIM * sizeof(float);
    float *u, *u_new;
    hiprandState *devStates;

    hipError_t err = hipMalloc((void **)&u, size);
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&u_new, size);
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&devStates, DIM * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    initialize<<<(DIM + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(devStates, u, 1234);
    hipDeviceSynchronize();

    float dt = 0.01f;
    float dx = 1.0f;

    for (int step = 0; step < 1000; ++step) {
        heatStep<<<(DIM + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(u, u_new, dt, dx);
        update<<<(DIM + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(u, u_new);
        hipDeviceSynchronize();
    }

    hipFree(u);
    hipFree(u_new);
    hipFree(devStates);

    return 0;
}