#include <iostream>
#include <hiprand/hiprand_kernel.h>
#include <hip/hip_runtime.h>

#define PATHS_PER_BLOCK 256
#define BLOCKS 16
#define PATHS (PATHS_PER_BLOCK * BLOCKS)
#define STEPS 256

__constant__ float d_params[4]; // [0] = S0, [1] = K, [2] = r, [3] = sigma

__global__ void monteCarloKernel(hiprandState *states, float *prices) {
    extern __shared__ float shared_prices[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    if (idx >= PATHS) return;

    hiprandState localState = states[idx];
    float S = d_params[0];
    float K = d_params[1];
    float r = d_params[2];
    float sigma = d_params[3];
    float dt = 1.0f / STEPS;
    float S_T = S;

    for (int step = 0; step < STEPS; step++) {
        float z = hiprand_normal(&localState);
        S_T *= exp((r - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * z);
    }

    float payoff = fmaxf(0.0f, S_T - K);
    shared_prices[tid] = payoff;

    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_prices[tid] += shared_prices[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        prices[blockIdx.x] = shared_prices[0] / blockDim.x;
    }

    states[idx] = localState;
}

__global__ void initStates(hiprandState *states, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= PATHS) return;
    hiprand_init(seed, idx, 0, &states[idx]);
}

int main() {
    hiprandState *devStates;
    float *devPrices;
    float *hostPrices = new float[BLOCKS];

    hipMalloc((void **)&devStates, PATHS * sizeof(hiprandState));
    hipMalloc((void **)&devPrices, BLOCKS * sizeof(float));

    // Initialize parameters
    float params[4] = {100.0f, 100.0f, 0.05f, 0.2f};
    hipMemcpyToSymbol(HIP_SYMBOL(d_params), params, 4 * sizeof(float));

    // Initialize random states
    initStates<<<BLOCKS, PATHS_PER_BLOCK>>>(devStates, 1234);
    hipDeviceSynchronize();

    // Run Monte Carlo simulation
    monteCarloKernel<<<BLOCKS, PATHS_PER_BLOCK, PATHS_PER_BLOCK * sizeof(float)>>>(devStates, devPrices);
    hipDeviceSynchronize();

    // Copy results back to host
    hipMemcpy(hostPrices, devPrices, BLOCKS * sizeof(float), hipMemcpyDeviceToHost);

    // Calculate average price
    float totalPrice = 0.0f;
    for (int i = 0; i < BLOCKS; i++) {
        totalPrice += hostPrices[i];
    }
    float averagePrice = totalPrice / BLOCKS;

    std::cout << "Average option price: " << averagePrice << std::endl;

    // Free device memory
    hipFree(devStates);
    hipFree(devPrices);
    delete[] hostPrices;

    return 0;
}