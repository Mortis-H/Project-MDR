// genetic_algorithm.cu
// A CUDA implementation of a genetic algorithm with a population size of 64.
// Optimized for memory bandwidth on the Hopper architecture.

#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 64
#define CHROMOSOME_LENGTH 32
#define GENERATIONS 1000

// Kernel to initialize the population with random values
__global__ void initializePopulation(hiprandState *states, unsigned int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        unsigned int chromosome = 0;
        for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
            chromosome = (chromosome << 1) | (hiprand(&states[idx]) & 1);
        }
        population[idx] = chromosome;
    }
}

// Kernel to perform crossover and mutation
__global__ void evolvePopulation(hiprandState *states, unsigned int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        // Simple crossover: swap half of the bits with another random individual
        int partnerIdx = hiprand(&states[idx]) % POPULATION_SIZE;
        unsigned int partner = population[partnerIdx];
        unsigned int mask = (1 << (CHROMOSOME_LENGTH / 2)) - 1;
        unsigned int newChromosome = (population[idx] & (mask << (CHROMOSOME_LENGTH / 2))) |
                                    (partner & mask);

        // Simple mutation: flip a bit with a small probability
        if (hiprand(&states[idx]) < 0.01f * UINT_MAX) {
            int bitPos = hiprand(&states[idx]) % CHROMOSOME_LENGTH;
            newChromosome ^= (1 << bitPos);
        }

        population[idx] = newChromosome;
    }
}

// Kernel to evaluate fitness (simple example: count number of 1s)
__global__ void evaluateFitness(unsigned int *population, float *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        unsigned int chromosome = population[idx];
        float score = 0.0f;
        for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
            score += (chromosome & 1);
            chromosome >>= 1;
        }
        fitness[idx] = score / CHROMOSOME_LENGTH;
    }
}

int main() {
    unsigned int *d_population;
    float *d_fitness;
    hiprandState *d_states;

    // Allocate memory on the device
    hipError_t err = hipMalloc((void **)&d_population, POPULATION_SIZE * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate population: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_fitness, POPULATION_SIZE * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate fitness: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_states, POPULATION_SIZE * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate states: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize random number generator states
    unsigned int seed = 1234;
    int blockSize = 256;
    int numBlocks = (POPULATION_SIZE + blockSize - 1) / blockSize;
    initializePopulation<<<numBlocks, blockSize>>>(d_states, d_population);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initializePopulation: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Genetic algorithm loop
    for (int generation = 0; generation < GENERATIONS; ++generation) {
        evolvePopulation<<<numBlocks, blockSize>>>(d_states, d_population);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to launch evolvePopulation: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        evaluateFitness<<<numBlocks, blockSize>>>(d_population, d_fitness);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to launch evaluateFitness: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy results back to host for display
    unsigned int *h_population = new unsigned int[POPULATION_SIZE];
    float *h_fitness = new float[POPULATION_SIZE];
    hipMemcpy(h_population, d_population, POPULATION_SIZE * sizeof(unsigned int), hipMemcpyDeviceToHost);
    hipMemcpy(h_fitness, d_fitness, POPULATION_SIZE * sizeof(float), hipMemcpyDeviceToHost);

    // Print the best individual
    float bestFitness = 0.0f;
    unsigned int bestIndividual = 0;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        if (h_fitness[i] > bestFitness) {
            bestFitness = h_fitness[i];
            bestIndividual = h_population[i];
        }
    }
    std::cout << "Best individual: " << bestIndividual << " with fitness: " << bestFitness << std::endl;

    // Free device memory
    hipFree(d_population);
    hipFree(d_fitness);
    hipFree(d_states);

    // Free host memory
    delete[] h_population;
    delete[] h_fitness;

    return 0;
}