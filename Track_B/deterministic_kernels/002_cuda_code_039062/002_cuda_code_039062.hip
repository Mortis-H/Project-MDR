#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define BLOCK_SIZE 256
#define NUM_PARTICLES 1024
#define NUM_GPUS 2

// Kernel to initialize random positions for particles in 4D space
__global__ void initParticles(hiprandState *states, float4 *positions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        hiprandState localState = states[idx];
        positions[idx].x = hiprand_uniform(&localState) * 100.0f;
        positions[idx].y = hiprand_uniform(&localState) * 100.0f;
        positions[idx].z = hiprand_uniform(&localState) * 100.0f;
        positions[idx].w = hiprand_uniform(&localState) * 100.0f;
        states[idx] = localState;
    }
}

// Kernel to perform DFS on particles
__global__ void dfsKernel(float4 *positions, int *visited, int *stack, int *stackTop) {
    extern __shared__ float4 sharedPositions[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    if (idx < NUM_PARTICLES) {
        sharedPositions[tid] = positions[idx];
    }
    __syncthreads();

    // Only the first thread in each block will perform DFS
    if (tid == 0) {
        int top = atomicAdd(stackTop, 1);
        stack[top] = idx;
        visited[idx] = 1;

        while (top >= 0) {
            int current = stack[top];
            top--;

            // Simulate DFS by visiting neighbors (for simplicity, just mark all as visited)
            for (int i = 0; i < NUM_PARTICLES; ++i) {
                if (!visited[i]) {
                    visited[i] = 1;
                    top = atomicAdd(stackTop, 1);
                    stack[top] = i;
                }
            }
        }
    }
}

int main() {
    int numDevices;
    hipGetDeviceCount(&numDevices);
    if (numDevices < NUM_GPUS) {
        std::cerr << "Error: Not enough GPUs available." << std::endl;
        return -1;
    }

    float4 *d_positions[NUM_GPUS];
    int *d_visited[NUM_GPUS];
    int *d_stack[NUM_GPUS];
    int *d_stackTop[NUM_GPUS];
    hiprandState *d_states[NUM_GPUS];

    int particlesPerGPU = NUM_PARTICLES / NUM_GPUS;
    int stackSize = NUM_PARTICLES;

    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);

        hipMalloc((void**)&d_positions[i], particlesPerGPU * sizeof(float4));
        hipMalloc((void**)&d_visited[i], particlesPerGPU * sizeof(int));
        hipMalloc((void**)&d_stack[i], stackSize * sizeof(int));
        hipMalloc((void**)&d_stackTop[i], sizeof(int));
        hipMalloc((void**)&d_states[i], particlesPerGPU * sizeof(hiprandState));

        hipMemset(d_visited[i], 0, particlesPerGPU * sizeof(int));
        hipMemset(d_stackTop[i], 0, sizeof(int));

        // Initialize random states
        hiprandState *devStates = d_states[i];
        int seed = 1234;
        unsigned int offset = 1;
        initParticles<<<(particlesPerGPU + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(devStates, d_positions[i]);
        hipDeviceSynchronize();
    }

    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);

        dfsKernel<<<(particlesPerGPU + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float4)>>>(
            d_positions[i], d_visited[i], d_stack[i], d_stackTop[i]
        );
        hipDeviceSynchronize();

        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);
        hipFree(d_positions[i]);
        hipFree(d_visited[i]);
        hipFree(d_stack[i]);
        hipFree(d_stackTop[i]);
        hipFree(d_states[i]);
    }

    std::cout << "DFS completed successfully on " << NUM_GPUS << " GPUs." << std::endl;
    return 0;
}