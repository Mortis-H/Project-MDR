// genetic_algorithm.cu
// A simple genetic algorithm implemented in CUDA with a focus on divergence reduction.
// Target architecture: Turing

#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 64
#define GENOME_LENGTH 32
#define NUM_GENERATIONS 100

// Kernel to initialize the population with random values
__global__ void initializePopulation(hiprandState *states, int *population) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < POPULATION_SIZE) {
        // Initialize random number generator
        hiprand_init(1234, idx, 0, &states[idx]);
        // Initialize genome
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            population[idx * GENOME_LENGTH + i] = hiprand(&states[idx]) % 2;
        }
    }
}

// Kernel to perform crossover between pairs of individuals
__global__ void crossover(hiprandState *states, int *population, int *newPopulation) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < POPULATION_SIZE / 2) {
        int parent1 = idx;
        int parent2 = POPULATION_SIZE - idx - 1;
        int crossoverPoint = hiprand(&states[idx]) % GENOME_LENGTH;
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            if (i < crossoverPoint) {
                newPopulation[idx * GENOME_LENGTH + i] = population[parent1 * GENOME_LENGTH + i];
                newPopulation[parent2 * GENOME_LENGTH + i] = population[parent2 * GENOME_LENGTH + i];
            } else {
                newPopulation[idx * GENOME_LENGTH + i] = population[parent2 * GENOME_LENGTH + i];
                newPopulation[parent2 * GENOME_LENGTH + i] = population[parent1 * GENOME_LENGTH + i];
            }
        }
    }
}

// Kernel to perform mutation on the population
__global__ void mutate(hiprandState *states, int *population, float mutationRate) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < POPULATION_SIZE) {
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            if (hiprand_uniform(&states[idx]) < mutationRate) {
                population[idx * GENOME_LENGTH + i] = 1 - population[idx * GENOME_LENGTH + i];
            }
        }
    }
}

// Kernel to evaluate fitness (simple example: count number of 1s)
__global__ void evaluateFitness(int *population, float *fitness) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < POPULATION_SIZE) {
        int count = 0;
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            count += population[idx * GENOME_LENGTH + i];
        }
        fitness[idx] = static_cast<float>(count);
    }
}

int main() {
    // Allocate memory on host
    int *h_population = new int[POPULATION_SIZE * GENOME_LENGTH];
    float *h_fitness = new float[POPULATION_SIZE];
    int *h_newPopulation = new int[POPULATION_SIZE * GENOME_LENGTH];

    // Allocate memory on device
    int *d_population, *d_newPopulation;
    float *d_fitness;
    hiprandState *d_states;
    hipError_t err;

    err = hipMalloc(&d_population, POPULATION_SIZE * GENOME_LENGTH * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error allocating d_population: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_newPopulation, POPULATION_SIZE * GENOME_LENGTH * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error allocating d_newPopulation: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_fitness, POPULATION_SIZE * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating d_fitness: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_states, POPULATION_SIZE * sizeof(hiprandState));
    if (err != hipSuccess) { std::cerr << "Error allocating d_states: " << hipGetErrorString(err) << std::endl; return -1; }

    // Initialize population
    int blockSize = 256;
    int numBlocks = (POPULATION_SIZE + blockSize - 1) / blockSize;
    initializePopulation<<<numBlocks, blockSize>>>(d_states, d_population);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "Error in initializePopulation: " << hipGetErrorString(err) << std::endl; return -1; }

    // Genetic algorithm loop
    float mutationRate = 0.01f;
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        // Evaluate fitness
        evaluateFitness<<<numBlocks, blockSize>>>(d_population, d_fitness);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "Error in evaluateFitness: " << hipGetErrorString(err) << std::endl; return -1; }

        // Crossover
        crossover<<<numBlocks / 2, blockSize>>>(d_states, d_population, d_newPopulation);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "Error in crossover: " << hipGetErrorString(err) << std::endl; return -1; }

        // Mutate
        mutate<<<numBlocks, blockSize>>>(d_states, d_newPopulation, mutationRate);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "Error in mutate: " << hipGetErrorString(err) << std::endl; return -1; }

        // Swap populations
        int *temp = d_population;
        d_population = d_newPopulation;
        d_newPopulation = temp;
    }

    // Copy final population back to host
    err = hipMemcpy(h_population, d_population, POPULATION_SIZE * GENOME_LENGTH * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "Error copying population to host: " << hipGetErrorString(err) << std::endl; return -1; }

    // Free device memory
    hipFree(d_population);
    hipFree(d_newPopulation);
    hipFree(d_fitness);
    hipFree(d_states);

    // Output the best individual
    int bestIndex = 0;
    float bestFitness = h_fitness[0];
    for (int i = 1; i < POPULATION_SIZE; ++i) {
        if (h_fitness[i] > bestFitness) {
            bestFitness = h_fitness[i];
            bestIndex = i;
        }
    }
    std::cout << "Best individual has fitness " << bestFitness << " with genome: ";
    for (int i = 0; i < GENOME_LENGTH; ++i) {
        std::cout << h_population[bestIndex * GENOME_LENGTH + i];
    }
    std::cout << std::endl;

    // Free host memory
    delete[] h_population;
    delete[] h_fitness;
    delete[] h_newPopulation;

    return 0;
}