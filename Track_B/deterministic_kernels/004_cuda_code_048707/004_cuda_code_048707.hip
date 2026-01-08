#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PATHS 32
#define NUM_STEPS 1024
#define BLOCK_SIZE 256

__global__ void initializeRandomStates(hiprandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PATHS) {
        hiprand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void simulatePaths(hiprandState *states, float *prices, float S0, float mu, float sigma, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PATHS) {
        float S = S0;
        hiprandState localState = states[idx];
        for (int step = 0; step < NUM_STEPS; step++) {
            float z = hiprand_normal(&localState);
            S *= exp((mu - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * z);
        }
        prices[idx] = S;
        states[idx] = localState;
    }
}

__global__ void calculatePayoffs(float *prices, float K, float *payoffs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PATHS) {
        payoffs[idx] = fmaxf(0.0f, prices[idx] - K);
    }
}

__global__ void calculateOptionPrice(float *payoffs, float *optionPrice) {
    __shared__ float sharedPayoffs[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PATHS) {
        sharedPayoffs[threadIdx.x] = payoffs[idx];
    } else {
        sharedPayoffs[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedPayoffs[threadIdx.x] += sharedPayoffs[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(optionPrice, sharedPayoffs[0]);
    }
}

int main() {
    float S0 = 100.0f; // Initial stock price
    float K = 100.0f;  // Strike price
    float T = 1.0f;    // Time to maturity
    float r = 0.05f;   // Risk-free rate
    float sigma = 0.2f; // Volatility
    float dt = T / NUM_STEPS;
    float mu = r - 0.5f * sigma * sigma;

    hiprandState *devStates;
    float *devPrices, *devPayoffs, *devOptionPrice;
    float optionPrice = 0.0f;

    hipError_t err;

    // Allocate memory on the device
    err = hipMalloc((void **)&devStates, NUM_PATHS * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for states: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMalloc((void **)&devPrices, NUM_PATHS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for prices: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMalloc((void **)&devPayoffs, NUM_PATHS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for payoffs: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMalloc((void **)&devOptionPrice, sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for option price: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // Initialize random states
    initializeRandomStates<<<(NUM_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(devStates, 1234);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching kernel initializeRandomStates: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // Simulate paths
    simulatePaths<<<(NUM_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(devStates, devPrices, S0, mu, sigma, dt);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching kernel simulatePaths: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // Calculate payoffs
    calculatePayoffs<<<(NUM_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(devPrices, K, devPayoffs);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching kernel calculatePayoffs: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // Set option price to zero
    err = hipMemset(devOptionPrice, 0, sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error setting device memory for option price: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // Calculate option price
    calculateOptionPrice<<<(NUM_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(devPayoffs, devOptionPrice);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching kernel calculateOptionPrice: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy result back to host
    err = hipMemcpy(&optionPrice, devOptionPrice, sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying result from device to host: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // Normalize the option price
    optionPrice /= NUM_PATHS;
    optionPrice *= expf(-r * T);

    // Output the result
    std::cout << "Estimated option price: " << optionPrice << std::endl;

    // Free device memory
    hipFree(devStates);
    hipFree(devPrices);
    hipFree(devPayoffs);
    hipFree(devOptionPrice);

    return 0;
}