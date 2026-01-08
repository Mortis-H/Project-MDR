#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define block size
#define BLOCK_SIZE 32

// Kernel to initialize random states
__global__ void initRandomStates(hiprandState *state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    hiprand_init(seed, id, 0, &state[id]);
}

// Kernel to perform Monte Carlo simulation for option pricing
__global__ void monteCarloSimulation(hiprandState *state, float *prices, float S0, float r, float sigma, float T, int steps) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float S = S0;
    float dt = T / steps;

    for (int i = 0; i < steps; i++) {
        float epsilon = hiprand_normal(&state[id]);
        S *= exp((r - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * epsilon);
    }

    prices[id] = fmaxf(0.0f, S - 100.0f); // Example payoff: European call option with strike 100
}

int main() {
    const int numPaths = 32;
    const int blockSize = BLOCK_SIZE;
    const int numBlocks = (numPaths + blockSize - 1) / blockSize;

    // Device pointers
    hiprandState *d_state;
    float *d_prices;

    // Allocate memory on device
    hipError_t err = hipMalloc(&d_state, numPaths * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for random states: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc(&d_prices, numPaths * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for prices: " << hipGetErrorString(err) << std::endl;
        hipFree(d_state);
        return -1;
    }

    // Initialize random states
    initRandomStates<<<numBlocks, blockSize>>>(d_state, 1234);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching initRandomStates kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_state);
        hipFree(d_prices);
        return -1;
    }

    // Parameters for option pricing
    float S0 = 100.0f; // Initial stock price
    float r = 0.05f;   // Risk-free rate
    float sigma = 0.2f; // Volatility
    float T = 1.0f;    // Time to maturity
    int steps = 100;   // Number of steps in the simulation

    // Allocate pinned (page-locked) memory for asynchronous data transfer
    float *h_prices;
    err = hipHostAlloc(&h_prices, numPaths * sizeof(float), hipHostMallocDefault);
    if (err != hipSuccess) {
        std::cerr << "Error allocating pinned memory for prices: " << hipGetErrorString(err) << std::endl;
        hipFree(d_state);
        hipFree(d_prices);
        return -1;
    }

    // Perform Monte Carlo simulation
    monteCarloSimulation<<<numBlocks, blockSize>>>(d_state, d_prices, S0, r, sigma, T, steps);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching monteCarloSimulation kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_state);
        hipFree(d_prices);
        hipHostFree(h_prices);
        return -1;
    }

    // Asynchronous copy of results from device to host
    hipMemcpyAsync(h_prices, d_prices, numPaths * sizeof(float), hipMemcpyDeviceToHost, 0);
    hipDeviceSynchronize(); // Wait for the asynchronous copy to complete

    // Calculate the average price (option price)
    float sum = 0.0f;
    for (int i = 0; i < numPaths; i++) {
        sum += h_prices[i];
    }
    float optionPrice = sum / numPaths;

    // Output the result
    std::cout << "Estimated option price: " << optionPrice << std::endl;

    // Free resources
    hipFree(d_state);
    hipFree(d_prices);
    hipHostFree(h_prices);

    return 0;
}