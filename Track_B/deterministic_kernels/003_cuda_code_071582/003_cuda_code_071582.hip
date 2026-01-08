#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <vector>
#include <algorithm>

#define NUM_PARTICLES 32
#define NUM_GPUS 2
#define DIMENSIONS 1
#define MAX_ITERATIONS 100
#define INERTIA 0.729
#define COGNITIVE_COEFF 1.49445
#define SOCIAL_COEFF 1.49445

struct Particle {
    float position[DIMENSIONS];
    float velocity[DIMENSIONS];
    float best_position[DIMENSIONS];
    float best_value;
};

__device__ float fitness_function(float position[DIMENSIONS]) {
    // Simple example: sphere function
    return position[0] * position[0];
}

__global__ void update_particles(Particle* particles, Particle* global_best, hiprandState* states, int num_particles) {
    int idx = threadIdx.x;
    if (idx < num_particles) {
        hiprandState localState = states[idx];
        for (int d = 0; d < DIMENSIONS; ++d) {
            float r1 = hiprand_uniform(&localState);
            float r2 = hiprand_uniform(&localState);
            particles[idx].velocity[d] = INERTIA * particles[idx].velocity[d] +
                                          COGNITIVE_COEFF * r1 * (particles[idx].best_position[d] - particles[idx].position[d]) +
                                          SOCIAL_COEFF * r2 * (global_best->position[d] - particles[idx].position[d]);
            particles[idx].position[d] += particles[idx].velocity[d];
        }
        states[idx] = localState;

        float new_value = fitness_function(particles[idx].position);
        if (new_value < particles[idx].best_value) {
            particles[idx].best_value = new_value;
            for (int d = 0; d < DIMENSIONS; ++d) {
                particles[idx].best_position[d] = particles[idx].position[d];
            }
        }
    }
}

__global__ void find_global_best(Particle* particles, Particle* global_best, int num_particles) {
    extern __shared__ Particle shared_particles[];

    int idx = threadIdx.x;
    if (idx < num_particles) {
        shared_particles[idx] = particles[idx];
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (idx < s && shared_particles[idx].best_value > shared_particles[idx + s].best_value) {
            shared_particles[idx] = shared_particles[idx + s];
        }
        __syncthreads();
    }

    if (idx == 0 && shared_particles[0].best_value < global_best->best_value) {
        *global_best = shared_particles[0];
    }
}

__global__ void initialize_particles(Particle* particles, hiprandState* states, int num_particles) {
    int idx = threadIdx.x;
    if (idx < num_particles) {
        hiprandState localState = states[idx];
        for (int d = 0; d < DIMENSIONS; ++d) {
            particles[idx].position[d] = hiprand_uniform(&localState) * 100.0f - 50.0f;
            particles[idx].velocity[d] = hiprand_uniform(&localState) * 2.0f - 1.0f;
            particles[idx].best_position[d] = particles[idx].position[d];
        }
        particles[idx].best_value = fitness_function(particles[idx].position);
        states[idx] = localState;
    }
}

__global__ void init_curand_states(hiprandState* states, unsigned long seed, int num_particles) {
    int idx = threadIdx.x;
    if (idx < num_particles) {
        hiprand_init(seed, idx, 0, &states[idx]);
    }
}

int main() {
    hipSetDevice(0);
    Particle* d_particles[NUM_GPUS];
    Particle* d_global_best[NUM_GPUS];
    hiprandState* d_states[NUM_GPUS];
    Particle h_global_best[NUM_GPUS];
    std::vector<hipStream_t> streams(NUM_GPUS);
    std::vector<hipDeviceProp_t> deviceProps(NUM_GPUS);

    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);
        hipGetDeviceProperties(&deviceProps[i], i);
        hipStreamCreate(&streams[i]);
        hipMalloc(&d_particles[i], NUM_PARTICLES * sizeof(Particle));
        hipMalloc(&d_global_best[i], sizeof(Particle));
        hipMalloc(&d_states[i], NUM_PARTICLES * sizeof(hiprandState));
        init_curand_states<<<1, NUM_PARTICLES, 0, streams[i]>>>(d_states[i], 1234, NUM_PARTICLES);
        hipDeviceSynchronize();
        initialize_particles<<<1, NUM_PARTICLES, 0, streams[i]>>>(d_particles[i], d_states[i], NUM_PARTICLES);
        hipDeviceSynchronize();
        hipMemcpy(&h_global_best[i], d_particles[i], sizeof(Particle), hipMemcpyDeviceToHost);
    }

    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        for (int i = 0; i < NUM_GPUS; ++i) {
            hipSetDevice(i);
            update_particles<<<1, NUM_PARTICLES, 0, streams[i]>>>(d_particles[i], d_global_best[i], d_states[i], NUM_PARTICLES);
            find_global_best<<<1, NUM_PARTICLES, NUM_PARTICLES * sizeof(Particle), streams[i]>>>(d_particles[i], d_global_best[i], NUM_PARTICLES);
            hipMemcpy(&h_global_best[i], d_global_best[i], sizeof(Particle), hipMemcpyDeviceToHost);
        }

        // Synchronize global best across GPUs
        Particle global_best = h_global_best[0];
        for (int i = 1; i < NUM_GPUS; ++i) {
            if (h_global_best[i].best_value < global_best.best_value) {
                global_best = h_global_best[i];
            }
        }

        for (int i = 0; i < NUM_GPUS; ++i) {
            hipSetDevice(i);
            hipMemcpy(d_global_best[i], &global_best, sizeof(Particle), hipMemcpyHostToDevice);
        }
    }

    // Print the best solution found
    Particle best_solution = h_global_best[0];
    for (int i = 1; i < NUM_GPUS; ++i) {
        if (h_global_best[i].best_value < best_solution.best_value) {
            best_solution = h_global_best[i];
        }
    }
    std::cout << "Best solution: " << best_solution.position[0] << " with value: " << best_solution.best_value << std::endl;

    // Cleanup
    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);
        hipFree(d_particles[i]);
        hipFree(d_global_best[i]);
        hipFree(d_states[i]);
        hipStreamDestroy(streams[i]);
    }

    return 0;
}