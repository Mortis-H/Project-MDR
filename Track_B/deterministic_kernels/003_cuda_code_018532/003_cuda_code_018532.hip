#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PATHS 16384
#define NUM_STEPS 100
#define BLOCK_SIZE 256

__global__ void initCurandStates(hiprandState *states, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_PATHS) {
        hiprand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void monteCarloSimulation(hiprandState *states, float *prices, float strike, float vol, float dt, float S0) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_PATHS) {
        float S = S0;
        hiprandState localState = states[idx];
        for (int step = 0; step < NUM_STEPS; ++step) {
            float epsilon = hiprand_normal(&localState);
            S *= exp((0.5f * vol * vol - 0.5f) * dt + vol * sqrtf(dt) * epsilon);
        }
        prices[idx] = fmaxf(0.0f, S - strike);
        states[idx] = localState;
    }
}

__global__ void calculatePayoff(float *prices, float *payoff) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        float sum = 0.0f;
        for (int i = 0; i < NUM_PATHS; ++i) {
            sum += prices[i];
        }
        *payoff = sum / NUM_PATHS;
    }
}

int main() {
    hiprandState *devStates;
    float *devPrices, *devPayoff;
    float strike = 100.0f;
    float vol = 0.2f;
    float dt = 0.01f;
    float S0 = 100.0f;
    float payoff = 0.0f;

    // Allocate Unified Memory
    hipMallocManaged(&devStates, NUM_PATHS * sizeof(hiprandState));
    hipMallocManaged(&devPrices, NUM_PATHS * sizeof(float));
    hipMallocManaged(&devPayoff, sizeof(float));

    // Initialize random states
    initCurandStates<<<(NUM_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(devStates, 1234);
    hipDeviceSynchronize();

    // Perform Monte Carlo simulation
    monteCarloSimulation<<<(NUM_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(devStates, devPrices, strike, vol, dt, S0);
    hipDeviceSynchronize();

    // Calculate payoff
    calculatePayoff<<<1, 1>>>(devPrices, devPayoff);
    hipDeviceSynchronize();

    // Copy result back to host
    hipMemcpy(&payoff, devPayoff, sizeof(float), hipMemcpyDeviceToHost);

    // Output the result
    std::cout << "Estimated option price: " << payoff << std::endl;

    // Free Unified Memory
    hipFree(devStates);
    hipFree(devPrices);
    hipFree(devPayoff);

    return 0;
}