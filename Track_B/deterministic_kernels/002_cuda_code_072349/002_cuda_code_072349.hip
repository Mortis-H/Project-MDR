#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <hip/hip_fp16.h>

#define PATHS_PER_BLOCK 32
#define BLOCKS 2
#define PATHS (PATHS_PER_BLOCK * BLOCKS)
#define STEPS 100
#define DT 0.01f
#define R 0.05f
#define SIGMA 0.2f
#define S0 100.0f
#define K 100.0f

__global__ void initCurandStates(hiprandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    hiprand_init(seed, idx, 0, &states[idx]);
}

__global__ void monteCarloKernel(hiprandState *states, float *prices, float *sumPayoffs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    hiprandState localState = states[idx];

    float S = S0;
    for (int step = 0; step < STEPS; ++step) {
        float epsilon = hiprand_normal(&localState);
        S *= exp((R - 0.5f * SIGMA * SIGMA) * DT + SIGMA * sqrtf(DT) * epsilon);
    }

    float payoff = fmaxf(0.0f, S - K);
    atomicAdd(&sumPayoffs[0], payoff);

    states[idx] = localState;
}

int main() {
    hiprandState *devStates;
    float *devPrices, *devSumPayoffs;
    float *hostSumPayoffs = new float[1];
    *hostSumPayoffs = 0.0f;

    size_t statesSize = PATHS * sizeof(hiprandState);
    size_t pricesSize = PATHS * sizeof(float);
    size_t sumPayoffsSize = sizeof(float);

    hipError_t err = hipMalloc(&devStates, statesSize);
    if (err != hipSuccess) {
        std::cerr << "Error allocating states: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc(&devPrices, pricesSize);
    if (err != hipSuccess) {
        std::cerr << "Error allocating prices: " << hipGetErrorString(err) << std::endl;
        hipFree(devStates);
        return -1;
    }

    err = hipMalloc(&devSumPayoffs, sumPayoffsSize);
    if (err != hipSuccess) {
        std::cerr << "Error allocating sumPayoffs: " << hipGetErrorString(err) << std::endl;
        hipFree(devStates);
        hipFree(devPrices);
        return -1;
    }

    err = hipMemcpy(devSumPayoffs, hostSumPayoffs, sumPayoffsSize, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying sumPayoffs to device: " << hipGetErrorString(err) << std::endl;
        hipFree(devStates);
        hipFree(devPrices);
        hipFree(devSumPayoffs);
        return -1;
    }

    dim3 threads(PATHS_PER_BLOCK);
    dim3 blocks(BLOCKS);

    initCurandStates<<<blocks, threads>>>(devStates, 12345);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing hiprand states: " << hipGetErrorString(err) << std::endl;
        hipFree(devStates);
        hipFree(devPrices);
        hipFree(devSumPayoffs);
        return -1;
    }

    monteCarloKernel<<<blocks, threads>>>(devStates, devPrices, devSumPayoffs);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching monteCarloKernel: " << hipGetErrorString(err) << std::endl;
        hipFree(devStates);
        hipFree(devPrices);
        hipFree(devSumPayoffs);
        return -1;
    }

    err = hipMemcpy(hostSumPayoffs, devSumPayoffs, sumPayoffsSize, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying sumPayoffs from device: " << hipGetErrorString(err) << std::endl;
        hipFree(devStates);
        hipFree(devPrices);
        hipFree(devSumPayoffs);
        return -1;
    }

    float optionPrice = (*hostSumPayoffs) / PATHS * exp(-R * STEPS * DT);
    std::cout << "Option price: " << optionPrice << std::endl;

    hipFree(devStates);
    hipFree(devPrices);
    hipFree(devSumPayoffs);
    delete[] hostSumPayoffs;

    return 0;
}