// simulated_annealing.cu
#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_STATES 128
#define BLOCK_SIZE 32

// Kernel to perform simulated annealing
__global__ void simulatedAnnealingKernel(hiprandState *states, float *energies, float *temperatures, float *bestEnergies, int *bestStates) {
    extern __shared__ float sharedEnergies[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    // Initialize random state
    hiprandState localState = states[idx];
    float energy = energies[idx];
    float temperature = temperatures[idx];
    int bestState = idx;

    // Perform simulated annealing
    for (int step = 0; step < 1000; ++step) {
        // Generate a random neighbor state
        int neighborState = (idx + hiprand(&localState) % (NUM_STATES - 1) + 1) % NUM_STATES;

        // Calculate the energy difference
        float deltaE = energies[neighborState] - energy;

        // Metropolis criterion
        if (deltaE < 0 || expf(-deltaE / temperature) > hiprand_uniform(&localState)) {
            idx = neighborState;
            energy = energies[idx];
        }

        // Update best state
        if (energy < bestEnergies[bid]) {
            bestEnergies[bid] = energy;
            bestStates[bid] = idx;
        }
    }

    // Store the best energy and state found by this block
    sharedEnergies[tid] = bestEnergies[bid];
    __syncthreads();

    // Find the best energy and state in the block
    if (tid == 0) {
        float minEnergy = sharedEnergies[0];
        int minIdx = 0;
        for (int i = 1; i < blockDim.x; ++i) {
            if (sharedEnergies[i] < minEnergy) {
                minEnergy = sharedEnergies[i];
                minIdx = i;
            }
        }
        bestEnergies[bid] = minEnergy;
        bestStates[bid] = bestStates[bid * blockDim.x + minIdx];
    }

    // Write back the state
    states[idx] = localState;
}

// Host function to initialize random states
__global__ void initRandomStates(hiprandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    hiprand_init(seed, idx, 0, &states[idx]);
}

int main() {
    int numBlocks = (NUM_STATES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate memory on host
    float *h_energies = new float[NUM_STATES];
    float *h_temperatures = new float[NUM_STATES];
    float *h_bestEnergies = new float[numBlocks];
    int *h_bestStates = new int[numBlocks];

    // Initialize energies and temperatures on host
    for (int i = 0; i < NUM_STATES; ++i) {
        h_energies[i] = static_cast<float>(rand()) / RAND_MAX; // Random initial energy
        h_temperatures[i] = 10.0f; // Initial temperature
    }

    // Allocate memory on device
    float *d_energies, *d_temperatures, *d_bestEnergies;
    int *d_bestStates;
    hiprandState *d_states;

    hipError_t err;
    err = hipMalloc((void**)&d_energies, NUM_STATES * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating d_energies: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_temperatures, NUM_STATES * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating d_temperatures: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_bestEnergies, numBlocks * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating d_bestEnergies: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_bestStates, numBlocks * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error allocating d_bestStates: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_states, NUM_STATES * sizeof(hiprandState));
    if (err != hipSuccess) { std::cerr << "Error allocating d_states: " << hipGetErrorString(err) << std::endl; return -1; }

    // Copy data from host to device
    err = hipMemcpy(d_energies, h_energies, NUM_STATES * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "Error copying h_energies to d_energies: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMemcpy(d_temperatures, h_temperatures, NUM_STATES * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "Error copying h_temperatures to d_temperatures: " << hipGetErrorString(err) << std::endl; return -1; }

    // Initialize random states on device
    initRandomStates<<<numBlocks, BLOCK_SIZE>>>(d_states, 12345);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "Error initializing random states: " << hipGetErrorString(err) << std::endl; return -1; }

    // Set initial best energies and states
    for (int i = 0; i < numBlocks; ++i) {
        h_bestEnergies[i] = 1e30f;
        h_bestStates[i] = 0;
    }
    err = hipMemcpy(d_bestEnergies, h_bestEnergies, numBlocks * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "Error copying h_bestEnergies to d_bestEnergies: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMemcpy(d_bestStates, h_bestStates, numBlocks * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "Error copying h_bestStates to d_bestStates: " << hipGetErrorString(err) << std::endl; return -1; }

    // Launch kernel
    simulatedAnnealingKernel<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_states, d_energies, d_temperatures, d_bestEnergies, d_bestStates);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "Error launching kernel: " << hipGetErrorString(err) << std::endl; return -1; }

    // Copy results back to host
    err = hipMemcpy(h_bestEnergies, d_bestEnergies, numBlocks * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "Error copying d_bestEnergies to h_bestEnergies: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMemcpy(h_bestStates, d_bestStates, numBlocks * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "Error copying d_bestStates to h_bestStates: " << hipGetErrorString(err) << std::endl; return -1; }

    // Find the best energy and state across all blocks
    float bestEnergy = h_bestEnergies[0];
    int bestState = h_bestStates[0];
    for (int i = 1; i < numBlocks; ++i) {
        if (h_bestEnergies[i] < bestEnergy) {
            bestEnergy = h_bestEnergies[i];
            bestState = h_bestStates[i];
        }
    }

    // Print the best energy and state
    std::cout << "Best energy: " << bestEnergy << " at state: " << bestState << std::endl;

    // Free device memory
    hipFree(d_energies);
    hipFree(d_temperatures);
    hipFree(d_bestEnergies);
    hipFree(d_bestStates);
    hipFree(d_states);

    // Free host memory
    delete[] h_energies;
    delete[] h_temperatures;
    delete[] h_bestEnergies;
    delete[] h_bestStates;

    return 0;
}