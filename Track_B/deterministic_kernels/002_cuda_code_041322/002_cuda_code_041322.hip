// genetic_algorithm.cu
#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 1024
#define CHROMOSOME_SIZE 32
#define NUM_GENERATIONS 100

__constant__ float crossover_rate = 0.7f;
__constant__ float mutation_rate = 0.01f;

__global__ void initializePopulation(hiprandState *states, int *population) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init(idx, 0, 0, &states[idx]);
        for (int i = 0; i < CHROMOSOME_SIZE; ++i) {
            population[idx * CHROMOSOME_SIZE + i] = hiprand(&states[idx]) % 2;
        }
    }
}

__device__ int evaluateFitness(int *chromosome) {
    int fitness = 0;
    for (int i = 0; i < CHROMOSOME_SIZE; ++i) {
        fitness += chromosome[i];
    }
    return fitness;
}

__device__ void crossover(int *parent1, int *parent2, int *child1, int *child2, hiprandState *state) {
    int crossover_point = hiprand(state) % CHROMOSOME_SIZE;
    for (int i = 0; i < CHROMOSOME_SIZE; ++i) {
        if (i < crossover_point) {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        } else {
            child1[i] = parent2[i];
            child2[i] = parent1[i];
        }
    }
}

__device__ void mutate(int *chromosome, hiprandState *state) {
    for (int i = 0; i < CHROMOSOME_SIZE; ++i) {
        if (hiprand_uniform(state) < mutation_rate) {
            chromosome[i] = 1 - chromosome[i];
        }
    }
}

__global__ void geneticAlgorithmStep(hiprandState *states, int *population, int *new_population) {
    extern __shared__ int shared_population[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < POPULATION_SIZE) {
        // Load data into shared memory for faster access
        for (int i = 0; i < CHROMOSOME_SIZE; ++i) {
            shared_population[threadIdx.x * CHROMOSOME_SIZE + i] = population[idx * CHROMOSOME_SIZE + i];
        }
        __syncthreads();

        // Evaluate fitness
        int fitness = evaluateFitness(&shared_population[threadIdx.x * CHROMOSOME_SIZE]);

        // Selection: Simple roulette wheel selection
        int total_fitness = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            total_fitness += evaluateFitness(&shared_population[i * CHROMOSOME_SIZE]);
        }
        __syncthreads();

        int target_fitness = hiprand(&states[idx]) % total_fitness;
        int accumulated_fitness = 0;
        int parent1_idx = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            accumulated_fitness += evaluateFitness(&shared_population[i * CHROMOSOME_SIZE]);
            if (accumulated_fitness >= target_fitness) {
                parent1_idx = i;
                break;
            }
        }
        __syncthreads();

        target_fitness = hiprand(&states[idx]) % total_fitness;
        accumulated_fitness = 0;
        int parent2_idx = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            accumulated_fitness += evaluateFitness(&shared_population[i * CHROMOSOME_SIZE]);
            if (accumulated_fitness >= target_fitness) {
                parent2_idx = i;
                break;
            }
        }
        __syncthreads();

        // Crossover and Mutation
        int *parent1 = &shared_population[parent1_idx * CHROMOSOME_SIZE];
        int *parent2 = &shared_population[parent2_idx * CHROMOSOME_SIZE];
        int *child1 = &new_population[idx * CHROMOSOME_SIZE];
        int *child2 = &new_population[(idx + blockDim.x / 2) % POPULATION_SIZE * CHROMOSOME_SIZE];

        if (hiprand_uniform(&states[idx]) < crossover_rate) {
            crossover(parent1, parent2, child1, child2, &states[idx]);
        } else {
            for (int i = 0; i < CHROMOSOME_SIZE; ++i) {
                child1[i] = parent1[i];
                child2[i] = parent2[i];
            }
        }

        mutate(child1, &states[idx]);
        mutate(child2, &states[idx]);
    }
}

int main() {
    int *h_population, *d_population, *d_new_population;
    hiprandState *d_states;

    // Allocate host memory
    h_population = (int *)malloc(POPULATION_SIZE * CHROMOSOME_SIZE * sizeof(int));

    // Allocate device memory
    hipError_t err = hipMalloc((void **)&d_population, POPULATION_SIZE * CHROMOSOME_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for population" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_new_population, POPULATION_SIZE * CHROMOSOME_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for new population" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_states, POPULATION_SIZE * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for states" << std::endl;
        return -1;
    }

    // Initialize population
    int blockSize = 256;
    int numBlocks = (POPULATION_SIZE + blockSize - 1) / blockSize;
    initializePopulation<<<numBlocks, blockSize>>>(d_states, d_population);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initializePopulation kernel" << std::endl;
        return -1;
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize after initializePopulation" << std::endl;
        return -1;
    }

    // Genetic Algorithm steps
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        geneticAlgorithmStep<<<numBlocks, blockSize, blockSize * CHROMOSOME_SIZE * sizeof(int)>>>(d_states, d_population, d_new_population);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to launch geneticAlgorithmStep kernel" << std::endl;
            return -1;
        }
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::cerr << "Failed to synchronize after geneticAlgorithmStep" << std::endl;
            return -1;
        }
        // Swap populations
        int *temp = d_population;
        d_population = d_new_population;
        d_new_population = temp;
    }

    // Copy result back to host
    err = hipMemcpy(h_population, d_population, POPULATION_SIZE * CHROMOSOME_SIZE * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy population from device to host" << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_population);
    hipFree(d_new_population);
    hipFree(d_states);

    // Free host memory
    free(h_population);

    return 0;
}