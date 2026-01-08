// genetic_algorithm.cu
#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Constants
#define POPULATION_SIZE 32
#define GENOME_LENGTH 64

// Kernel to initialize the population with random values
__global__ void initializePopulation(hiprandState *state, int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        // Initialize curand state
        hiprand_init(1234, idx, 0, &state[idx]);
        // Generate random genome
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            population[idx * GENOME_LENGTH + i] = hiprand(&state[idx]) % 2;
        }
    }
}

// Kernel to evaluate fitness (simple example: count number of 1s)
__global__ void evaluateFitness(int *population, int *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int fit = 0;
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            fit += population[idx * GENOME_LENGTH + i];
        }
        fitness[idx] = fit;
    }
}

// Main function to demonstrate kernel usage
int main() {
    // Allocate memory for population and fitness on host
    int *h_population = new int[POPULATION_SIZE * GENOME_LENGTH];
    int *h_fitness = new int[POPULATION_SIZE];

    // Allocate memory for population, fitness, and curand states on device
    int *d_population, *d_fitness;
    hiprandState *d_state;
    hipMalloc((void**)&d_population, POPULATION_SIZE * GENOME_LENGTH * sizeof(int));
    hipMalloc((void**)&d_fitness, POPULATION_SIZE * sizeof(int));
    hipMalloc((void**)&d_state, POPULATION_SIZE * sizeof(hiprandState));

    // Initialize population
    int blockSize = 32;
    int numBlocks = (POPULATION_SIZE + blockSize - 1) / blockSize;
    initializePopulation<<<numBlocks, blockSize>>>(d_state, d_population);
    hipDeviceSynchronize();

    // Check for errors in kernel launch
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Evaluate fitness
    evaluateFitness<<<numBlocks, blockSize>>>(d_population, d_fitness);
    hipDeviceSynchronize();

    // Check for errors in kernel launch
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    hipMemcpy(h_population, d_population, POPULATION_SIZE * GENOME_LENGTH * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_fitness, d_fitness, POPULATION_SIZE * sizeof(int), hipMemcpyDeviceToHost);

    // Output fitness results
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        std::cout << "Individual " << i << " Fitness: " << h_fitness[i] << std::endl;
    }

    // Free device memory
    hipFree(d_population);
    hipFree(d_fitness);
    hipFree(d_state);

    // Free host memory
    delete[] h_population;
    delete[] h_fitness;

    return 0;
}