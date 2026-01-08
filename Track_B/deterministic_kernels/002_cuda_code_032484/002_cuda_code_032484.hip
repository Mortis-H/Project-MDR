// monte_carlo_cuda.cu
// A CUDA implementation of a Monte Carlo simulation for option pricing using dynamic parallelism.

#include <iostream>
#include <hiprand/hiprand_kernel.h>
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 256
#define PATHS_PER_BLOCK 2
#define TOTAL_PATHS 512

// Kernel to simulate stock prices using the geometric Brownian motion model
__global__ void simulate_paths(float *d_prices, float S0, float mu, float sigma, float T, int steps) {
    int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    int paths_per_block = blockDim.x;

    // Allocate shared memory for paths
    extern __shared__ float s_paths[];

    // Initialize random number generator
    hiprandState localState;
    hiprand_init(threadIdx.x, 0, 0, &localState);

    // Initialize the stock price for this path
    float price = S0;
    s_paths[threadIdx.x] = price;

    // Simulate the stock price path
    for (int step = 0; step < steps; ++step) {
        float dt = T / steps;
        float z = hiprand_normal(&localState);
        price *= exp((mu - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * z);
        s_paths[threadIdx.x] = price;
    }

    // Store the final price in global memory
    if (threadIdx.x < paths_per_block) {
        d_prices[path_id] = s_paths[threadIdx.x];
    }
}

// Kernel to calculate the option price using the simulated stock prices
__global__ void calculate_option_price(float *d_prices, float K, float *d_option_price) {
    extern __shared__ float s_prices[];

    int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    int paths_per_block = blockDim.x;

    // Load the stock prices into shared memory
    s_prices[threadIdx.x] = d_prices[path_id];

    __syncthreads();

    // Calculate the payoff for each path
    float payoff = fmaxf(0.0f, s_prices[threadIdx.x] - K);

    // Perform reduction to calculate the average payoff
    for (int stride = paths_per_block / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_prices[threadIdx.x] += s_prices[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write the result to global memory
    if (threadIdx.x == 0) {
        atomicAdd(d_option_price, s_prices[0]);
    }
}

int main() {
    // Parameters for the option pricing
    float S0 = 100.0f; // Initial stock price
    float K = 100.0f;  // Strike price
    float mu = 0.05f;  // Drift
    float sigma = 0.2f; // Volatility
    float T = 1.0f;    // Time to maturity
    int steps = 100;   // Number of time steps

    // Allocate device memory
    float *d_prices, *d_option_price;
    hipMalloc((void **)&d_prices, TOTAL_PATHS * sizeof(float));
    hipMalloc((void **)&d_option_price, sizeof(float));
    hipMemset(d_option_price, 0, sizeof(float));

    // Launch the simulation kernel
    int blocks = TOTAL_PATHS / (BLOCK_SIZE * PATHS_PER_BLOCK);
    simulate_paths<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_prices, S0, mu, sigma, T, steps);
    hipDeviceSynchronize();

    // Launch the option price calculation kernel
    calculate_option_price<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_prices, K, d_option_price);
    hipDeviceSynchronize();

    // Copy the result back to the host
    float option_price;
    hipMemcpy(&option_price, d_option_price, sizeof(float), hipMemcpyDeviceToHost);

    // Normalize the option price
    option_price /= TOTAL_PATHS;
    option_price *= exp(-mu * T);

    // Output the result
    std::cout << "Estimated option price: " << option_price << std::endl;

    // Free device memory
    hipFree(d_prices);
    hipFree(d_option_price);

    return 0;
}