#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PATHS 128
#define NUM_STEPS 1024
#define BLOCK_SIZE 128

// Kernel to initialize random states
__global__ void initRandomStates(hiprandState *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_PATHS) {
        hiprand_init(seed, idx, 0, &state[idx]);
    }
}

// Kernel to perform Monte Carlo simulation
__global__ void monteCarloSimulation(hiprandState *state, float *prices) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_PATHS) {
        float S = 100.0f; // Initial stock price
        float r = 0.05f;  // Risk-free rate
        float sigma = 0.2f; // Volatility
        float dt = 1.0f / NUM_STEPS; // Time step
        float S_next;

        hiprandState localState = state[idx];
        for (int step = 0; step < NUM_STEPS; ++step) {
            float Z = hiprand_normal(&localState);
            S_next = S * exp((r - 0.5f * sigma * sigma) * dt + sigma * sqrt(dt) * Z);
            S = S_next;
        }
        state[idx] = localState;
        prices[idx] = S;
    }
}

int main() {
    hiprandState *d_state;
    float *d_prices, *h_prices;
    size_t stateSize = NUM_PATHS * sizeof(hiprandState);
    size_t priceSize = NUM_PATHS * sizeof(float);

    // Allocate memory on device
    hipError_t err = hipMalloc(&d_state, stateSize);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for states: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&d_prices, priceSize);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for prices: " << hipGetErrorString(err) << std::endl;
        hipFree(d_state);
        return -1;
    }

    // Allocate memory on host
    h_prices = (float *)malloc(priceSize);
    if (!h_prices) {
        std::cerr << "Failed to allocate memory for host prices" << std::endl;
        hipFree(d_state);
        hipFree(d_prices);
        return -1;
    }

    // Initialize random states on device
    initRandomStates<<<(NUM_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_state, 1234);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_state);
        hipFree(d_prices);
        free(h_prices);
        return -1;
    }

    // Perform Monte Carlo simulation
    monteCarloSimulation<<<(NUM_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_state, d_prices);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_state);
        hipFree(d_prices);
        free(h_prices);
        return -1;
    }

    // Copy results from device to host
    err = hipMemcpyAsync(h_prices, d_prices, priceSize, hipMemcpyDeviceToHost, 0);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy memory from device to host: " << hipGetErrorString(err) << std::endl;
        hipFree(d_state);
        hipFree(d_prices);
        free(h_prices);
        return -1;
    }

    // Synchronize to ensure all data is copied
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_state);
        hipFree(d_prices);
        free(h_prices);
        return -1;
    }

    // Output the final stock prices
    for (int i = 0; i < NUM_PATHS; ++i) {
        std::cout << "Path " << i << ": " << h_prices[i] << std::endl;
    }

    // Free device and host memory
    hipFree(d_state);
    hipFree(d_prices);
    free(h_prices);

    return 0;
}