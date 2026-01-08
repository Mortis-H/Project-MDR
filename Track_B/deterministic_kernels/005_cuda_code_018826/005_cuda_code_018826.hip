#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 2048
#define CHROMOSOME_LENGTH 32
#define GENERATIONS 100

__global__ void initializePopulation(hiprandState *states, unsigned int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init((unsigned long long) clock() + idx, 0, 0, &states[idx]);
        for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
            population[idx * CHROMOSOME_LENGTH + i] = hiprand(&states[idx]) % 2;
        }
    }
}

__device__ unsigned int fitness(unsigned int *chromosome) {
    unsigned int score = 0;
    for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
        score += chromosome[i];
    }
    return score;
}

__global__ void evaluatePopulation(unsigned int *population, unsigned int *fitnessScores) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        fitnessScores[idx] = fitness(&population[idx * CHROMOSOME_LENGTH]);
    }
}

__global__ void selectParents(unsigned int *population, unsigned int *fitnessScores, unsigned int *parents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        unsigned int maxFitness = 0;
        int maxIndex = 0;
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            if (fitnessScores[i] > maxFitness) {
                maxFitness = fitnessScores[i];
                maxIndex = i;
            }
        }
        parents[idx] = maxIndex;
    }
}

__global__ void crossover(unsigned int *population, unsigned int *parents, hiprandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprandState localState = states[idx];
        unsigned int parent1 = parents[idx];
        unsigned int parent2 = parents[(idx + 1) % POPULATION_SIZE];
        int crossoverPoint = hiprand(&localState) % CHROMOSOME_LENGTH;
        for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
            if (i < crossoverPoint) {
                population[idx * CHROMOSOME_LENGTH + i] = population[parent1 * CHROMOSOME_LENGTH + i];
            } else {
                population[idx * CHROMOSOME_LENGTH + i] = population[parent2 * CHROMOSOME_LENGTH + i];
            }
        }
        states[idx] = localState;
    }
}

__global__ void mutate(unsigned int *population, hiprandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprandState localState = states[idx];
        for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
            if (hiprand(&localState) % 100 < 5) { // 5% mutation rate
                population[idx * CHROMOSOME_LENGTH + i] = 1 - population[idx * CHROMOSOME_LENGTH + i];
            }
        }
        states[idx] = localState;
    }
}

int main() {
    unsigned int *d_population, *d_fitnessScores, *d_parents;
    hiprandState *d_states;

    hipError_t err;

    err = hipMalloc((void **)&d_population, POPULATION_SIZE * CHROMOSOME_LENGTH * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating population: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void **)&d_fitnessScores, POPULATION_SIZE * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating fitnessScores: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void **)&d_parents, POPULATION_SIZE * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating parents: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void **)&d_states, POPULATION_SIZE * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Error allocating states: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    int blockSize = 256;
    int numBlocks = (POPULATION_SIZE + blockSize - 1) / blockSize;

    initializePopulation<<<numBlocks, blockSize>>>(d_states, d_population);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing population: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    for (int generation = 0; generation < GENERATIONS; ++generation) {
        evaluatePopulation<<<numBlocks, blockSize>>>(d_population, d_fitnessScores);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error evaluating population: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        selectParents<<<numBlocks, blockSize>>>(d_population, d_fitnessScores, d_parents);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error selecting parents: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        crossover<<<numBlocks, blockSize>>>(d_population, d_parents, d_states);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error performing crossover: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        mutate<<<numBlocks, blockSize>>>(d_population, d_states);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error performing mutation: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    unsigned int *h_fitnessScores = new unsigned int[POPULATION_SIZE];
    hipMemcpy(h_fitnessScores, d_fitnessScores, POPULATION_SIZE * sizeof(unsigned int), hipMemcpyDeviceToHost);

    unsigned int maxFitness = 0;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        if (h_fitnessScores[i] > maxFitness) {
            maxFitness = h_fitnessScores[i];
        }
    }

    std::cout << "Best fitness: " << maxFitness << std::endl;

    delete[] h_fitnessScores;

    hipFree(d_population);
    hipFree(d_fitnessScores);
    hipFree(d_parents);
    hipFree(d_states);

    return 0;
}