#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 32
#define GENOME_LENGTH 1024

// Kernel to perform genetic algorithm operations
__global__ void geneticAlgorithmKernel(hiprandState *states, int *population, int *newPopulation) {
    int idx = threadIdx.x;
    hiprandState localState = states[idx];
    
    // Perform crossover
    int parent1 = hiprand(&localState) % POPULATION_SIZE;
    int parent2 = hiprand(&localState) % POPULATION_SIZE;
    int crossoverPoint = hiprand(&localState) % GENOME_LENGTH;

    for (int i = 0; i < GENOME_LENGTH; ++i) {
        if (i < crossoverPoint) {
            newPopulation[idx * GENOME_LENGTH + i] = population[parent1 * GENOME_LENGTH + i];
        } else {
            newPopulation[idx * GENOME_LENGTH + i] = population[parent2 * GENOME_LENGTH + i];
        }
    }

    // Perform mutation
    for (int i = 0; i < GENOME_LENGTH; ++i) {
        if (hiprand(&localState) < 0.01f) { // Mutation rate of 1%
            newPopulation[idx * GENOME_LENGTH + i] = !newPopulation[idx * GENOME_LENGTH + i];
        }
    }

    // Write back the state
    states[idx] = localState;
}

// Initialize random states for the curand library
__global__ void initRandomStates(hiprandState *states, unsigned long seed) {
    int idx = threadIdx.x;
    hiprand_init(seed, idx, 0, &states[idx]);
}

int main() {
    // Allocate memory for random states
    hiprandState *devStates;
    hipMalloc((void**)&devStates, POPULATION_SIZE * sizeof(hiprandState));

    // Allocate memory for population and new population
    int *devPopulation, *devNewPopulation;
    hipMalloc((void**)&devPopulation, POPULATION_SIZE * GENOME_LENGTH * sizeof(int));
    hipMalloc((void**)&devNewPopulation, POPULATION_SIZE * GENOME_LENGTH * sizeof(int));

    // Initialize population with random values
    int *hPopulation = new int[POPULATION_SIZE * GENOME_LENGTH];
    for (int i = 0; i < POPULATION_SIZE * GENOME_LENGTH; ++i) {
        hPopulation[i] = rand() % 2;
    }
    hipMemcpy(devPopulation, hPopulation, POPULATION_SIZE * GENOME_LENGTH * sizeof(int), hipMemcpyHostToDevice);

    // Initialize random states
    initRandomStates<<<1, POPULATION_SIZE>>>(devStates, time(NULL));
    hipDeviceSynchronize();

    // Launch genetic algorithm kernel
    geneticAlgorithmKernel<<<1, POPULATION_SIZE>>>(devStates, devPopulation, devNewPopulation);
    hipDeviceSynchronize();

    // Copy results back to host
    int *hNewPopulation = new int[POPULATION_SIZE * GENOME_LENGTH];
    hipMemcpy(hNewPopulation, devNewPopulation, POPULATION_SIZE * GENOME_LENGTH * sizeof(int), hipMemcpyDeviceToHost);

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(devStates);
    hipFree(devPopulation);
    hipFree(devNewPopulation);

    // Free host memory
    delete[] hPopulation;
    delete[] hNewPopulation;

    return 0;
}