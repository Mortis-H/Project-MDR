#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PATHS 128
#define NUM_STEPS 1000
#define BLOCK_SIZE 32

// Kernel to initialize random number generators
__global__ void initRNG(hiprandState *state, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < NUM_PATHS) {
        hiprand_init(seed, id, 0, &state[id]);
    }
}

// Kernel to simulate a single path
__global__ void simulatePath(hiprandState *state, float *prices, float S0, float mu, float sigma, float dt) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < NUM_PATHS) {
        float S = S0;
        hiprandState localState = state[id];
        for (int step = 0; step < NUM_STEPS; ++step) {
            float z = hiprand_normal(&localState);
            S *= exp((mu - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * z);
        }
        prices[id] = S;
        state[id] = localState;
    }
}

// Kernel to calculate the option price using the simulated paths
__global__ void calculateOptionPrice(float *prices, float *optionPrice, float K, float dt) {
    __shared__ float sharedPrices[BLOCK_SIZE];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < NUM_PATHS) {
        sharedPrices[threadIdx.x] = fmaxf(0.0f, prices[id] - K);
        __syncthreads();

        // Reduce the prices in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sharedPrices[threadIdx.x] += sharedPrices[threadIdx.x + s];
            }
            __syncthreads();
        }

        // Write the block result to global memory
        if (threadIdx.x == 0) {
            atomicAdd(optionPrice, sharedPrices[0]);
        }
    }
}

int main() {
    // Initialize device
    int device;
    hipGetDevice(&device);
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, device);
    if (!deviceProp.managedMemory) {
        std::cerr << "Device does not support Unified Memory" << std::endl;
        return -1;
    }

    // Allocate memory
    hiprandState *d_state;
    float *d_prices, *d_optionPrice;
    hipMalloc(&d_state, NUM_PATHS * sizeof(hiprandState));
    hipMalloc(&d_prices, NUM_PATHS * sizeof(float));
    hipMalloc(&d_optionPrice, sizeof(float));
    hipMemset(d_optionPrice, 0, sizeof(float));

    // Initialize random number generators
    initRNG<<<(NUM_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_state, 123456789);
    hipDeviceSynchronize();

    // Simulate paths
    float S0 = 100.0f; // Initial stock price
    float mu = 0.05f;  // Drift coefficient
    float sigma = 0.2f; // Volatility
    float T = 1.0f;    // Time to maturity
    float dt = T / NUM_STEPS; // Time step
    simulatePath<<<(NUM_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_state, d_prices, S0, mu, sigma, dt);
    hipDeviceSynchronize();

    // Calculate option price
    float K = 100.0f; // Strike price
    calculateOptionPrice<<<(NUM_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_prices, d_optionPrice, K, dt);
    hipDeviceSynchronize();

    // Copy result back to host
    float optionPrice;
    hipMemcpy(&optionPrice, d_optionPrice, sizeof(float), hipMemcpyDeviceToHost);
    optionPrice /= NUM_PATHS; // Average over all paths

    // Output result
    std::cout << "Estimated option price: " << optionPrice << std::endl;

    // Free memory
    hipFree(d_state);
    hipFree(d_prices);
    hipFree(d_optionPrice);

    return 0;
}