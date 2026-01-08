#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define the population size and other constants
#define POPULATION_SIZE 1024
#define GENES 32
#define MAX_GENERATIONS 100

// Kernel to initialize the population with random values
__global__ void initializePopulation(hiprandState *states, int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init(idx, 0, 0, &states[idx]);
        for (int i = 0; i < GENES; ++i) {
            population[idx * GENES + i] = hiprand(&states[idx]) % 2;
        }
    }
}

// Kernel to evaluate the fitness of each individual
__global__ void evaluateFitness(int *population, int *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int fit = 0;
        for (int i = 0; i < GENES; ++i) {
            fit += population[idx * GENES + i];
        }
        fitness[idx] = fit;
    }
}

// Kernel to perform selection based on fitness
__global__ void selection(int *population, int *fitness, int *newPopulation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int parent1 = idx % (POPULATION_SIZE / 2);
        int parent2 = (idx + 1) % (POPULATION_SIZE / 2);
        if (fitness[parent1] > fitness[parent2]) {
            for (int i = 0; i < GENES; ++i) {
                newPopulation[idx * GENES + i] = population[parent1 * GENES + i];
            }
        } else {
            for (int i = 0; i < GENES; ++i) {
                newPopulation[idx * GENES + i] = population[parent2 * GENES + i];
            }
        }
    }
}

// Kernel to perform crossover between pairs of individuals
__global__ void crossover(int *population, int *newPopulation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE / 2) {
        int crossoverPoint = idx % GENES;
        for (int i = 0; i < crossoverPoint; ++i) {
            newPopulation[idx * GENES + i] = population[idx * GENES + i];
            newPopulation[(POPULATION_SIZE - idx - 1) * GENES + i] = population[(POPULATION_SIZE - idx - 1) * GENES + i];
        }
        for (int i = crossoverPoint; i < GENES; ++i) {
            newPopulation[idx * GENES + i] = population[(POPULATION_SIZE - idx - 1) * GENES + i];
            newPopulation[(POPULATION_SIZE - idx - 1) * GENES + i] = population[idx * GENES + i];
        }
    }
}

// Kernel to perform mutation on the population
__global__ void mutate(int *population, hiprandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        for (int i = 0; i < GENES; ++i) {
            if (hiprand(&states[idx]) < 0.01f) {
                population[idx * GENES + i] = 1 - population[idx * GENES + i];
            }
        }
    }
}

// Main function to demonstrate the genetic algorithm
int main() {
    int *d_population, *d_newPopulation, *d_fitness;
    hiprandState *d_states;
    int *h_population = new int[POPULATION_SIZE * GENES];
    int *h_fitness = new int[POPULATION_SIZE];

    // Allocate memory on the device
    hipError_t err = hipMalloc((void **)&d_population, POPULATION_SIZE * GENES * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for population (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_newPopulation, POPULATION_SIZE * GENES * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for new population (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_fitness, POPULATION_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for fitness (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_states, POPULATION_SIZE * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for states (error code " << err << ")" << std::endl;
        return -1;
    }

    // Initialize population
    int blockSize = 256;
    int numBlocks = (POPULATION_SIZE + blockSize - 1) / blockSize;
    initializePopulation<<<numBlocks, blockSize>>>(d_states, d_population);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize population (error code " << err << ")" << std::endl;
        return -1;
    }

    // Genetic algorithm loop
    for (int generation = 0; generation < MAX_GENERATIONS; ++generation) {
        // Evaluate fitness
        evaluateFitness<<<numBlocks, blockSize>>>(d_population, d_fitness);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to evaluate fitness (error code " << err << ")" << std::endl;
            return -1;
        }

        // Selection
        selection<<<numBlocks, blockSize>>>(d_population, d_fitness, d_newPopulation);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to perform selection (error code " << err << ")" << std::endl;
            return -1;
        }

        // Crossover
        crossover<<<numBlocks / 2, blockSize>>>(d_newPopulation, d_population);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to perform crossover (error code " << err << ")" << std::endl;
            return -1;
        }

        // Mutate
        mutate<<<numBlocks, blockSize>>>(d_population, d_states);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to perform mutation (error code " << err << ")" << std::endl;
            return -1;
        }
    }

    // Copy the final population back to the host
    hipMemcpy(h_population, d_population, POPULATION_SIZE * GENES * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_fitness, d_fitness, POPULATION_SIZE * sizeof(int), hipMemcpyDeviceToHost);

    // Find the best individual
    int bestFitness = 0;
    int bestIndividual = 0;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        if (h_fitness[i] > bestFitness) {
            bestFitness = h_fitness[i];
            bestIndividual = i;
        }
    }

    // Output the best individual
    std::cout << "Best individual: ";
    for (int i = 0; i < GENES; ++i) {
        std::cout << h_population[bestIndividual * GENES + i];
    }
    std::cout << " with fitness " << bestFitness << std::endl;

    // Free device memory
    hipFree(d_population);
    hipFree(d_newPopulation);
    hipFree(d_fitness);
    hipFree(d_states);

    // Free host memory
    delete[] h_population;
    delete[] h_fitness;

    return 0;
}