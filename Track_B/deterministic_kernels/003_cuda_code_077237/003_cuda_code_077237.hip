#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 1024  // Grid size in each dimension
#define BLOCK_SIZE 32  // Block size for each dimension

__global__ void initialize_grid(float *grid, hiprandState *states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int index = idz * N * N + idy * N + idx;

    if (idx < N && idy < N && idz < N) {
        hiprandState localState = states[index];
        hiprand_init(seed, index, 0, &localState);
        grid[index] = hiprand_uniform(&localState);
        states[index] = localState;
    }
}

__global__ void heat_equation(float *grid, float *grid_new, float alpha, int iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int index = idz * N * N + idy * N + idx;

    if (idx > 0 && idx < N - 1 && idy > 0 && idy < N - 1 && idz > 0 && idz < N - 1) {
        float sum = grid[index - N * N] + grid[index + N * N] +  // z-dim
                    grid[index - N] + grid[index + N] +          // y-dim
                    grid[index - 1] + grid[index + 1];           // x-dim
        grid_new[index] = grid[index] + alpha * (sum - 6 * grid[index]);
    }
}

__global__ void copy_grid(float *grid, float *grid_new) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int index = idz * N * N + idy * N + idx;

    if (idx < N && idy < N && idz < N) {
        grid[index] = grid_new[index];
    }
}

int main() {
    float *d_grid, *d_grid_new;
    hiprandState *d_states;
    float alpha = 0.1f;
    int iter = 1000;

    size_t grid_size = N * N * N * sizeof(float);
    size_t states_size = N * N * N * sizeof(hiprandState);

    hipError_t err;

    err = hipMalloc(&d_grid, grid_size);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate d_grid (error code " << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    err = hipMalloc(&d_grid_new, grid_size);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate d_grid_new (error code " << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    err = hipMalloc(&d_states, states_size);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate d_states (error code " << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    initialize_grid<<<numBlocks, threadsPerBlock>>>(d_grid, d_states, time(NULL));
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initialize_grid kernel (error code " << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < iter; ++i) {
        heat_equation<<<numBlocks, threadsPerBlock>>>(d_grid, d_grid_new, alpha, i);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to launch heat_equation kernel (error code " << err << ")" << std::endl;
            exit(EXIT_FAILURE);
        }

        copy_grid<<<numBlocks, threadsPerBlock>>>(d_grid, d_grid_new);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to launch copy_grid kernel (error code " << err << ")" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    hipFree(d_grid);
    hipFree(d_grid_new);
    hipFree(d_states);

    return 0;
}