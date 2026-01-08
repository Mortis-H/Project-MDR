#include <iostream>
#include <hip/hip_runtime.h>

// Constants
#define NUM_PARTICLES 32
#define DIM 3
#define DT 0.001f
#define CUTOFF 1.0f
#define EPSILON 1.0f
#define SIGMA 1.0f

// Kernel to update positions using velocity Verlet
__global__ void update_positions(float* pos, float* vel, float* acc, int num_particles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        for (int d = 0; d < DIM; ++d) {
            pos[idx * DIM + d] += vel[idx * DIM + d] * dt + 0.5f * acc[idx * DIM + d] * dt * dt;
        }
    }
}

// Kernel to calculate forces and update velocities using velocity Verlet
__global__ void calculate_forces(float* pos, float* vel, float* acc, int num_particles, float dt) {
    extern __shared__ float shared_pos[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        // Load positions into shared memory
        for (int d = 0; d < DIM; ++d) {
            shared_pos[threadIdx.x * DIM + d] = pos[idx * DIM + d];
        }
        __syncthreads();

        float new_acc[DIM] = {0.0f, 0.0f, 0.0f};
        for (int j = 0; j < num_particles; ++j) {
            if (j != idx) {
                float r[DIM];
                float r2 = 0.0f;
                for (int d = 0; d < DIM; ++d) {
                    r[d] = shared_pos[j * DIM + d] - pos[idx * DIM + d];
                    r2 += r[d] * r[d];
                }
                if (r2 < CUTOFF * CUTOFF) {
                    float r6 = r2 * r2 * r2;
                    float force = 24.0f * EPSILON * (2.0f * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA / r6 / r6 / r6 - SIGMA * SIGMA * SIGMA * SIGMA * SIGMA / r6) / r2;
                    for (int d = 0; d < DIM; ++d) {
                        new_acc[d] += force * r[d];
                    }
                }
            }
        }
        __syncthreads();

        // Update velocities
        for (int d = 0; d < DIM; ++d) {
            vel[idx * DIM + d] += 0.5f * (acc[idx * DIM + d] + new_acc[d]) * dt;
            acc[idx * DIM + d] = new_acc[d];
        }
    }
}

int main() {
    // Allocate host memory
    float* h_pos = new float[NUM_PARTICLES * DIM];
    float* h_vel = new float[NUM_PARTICLES * DIM];
    float* h_acc = new float[NUM_PARTICLES * DIM];

    // Initialize positions, velocities, and accelerations
    for (int i = 0; i < NUM_PARTICLES * DIM; ++i) {
        h_pos[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
        h_vel[i] = 0.0f;
        h_acc[i] = 0.0f;
    }

    // Allocate device memory
    float* d_pos, *d_vel, *d_acc;
    hipMalloc(&d_pos, NUM_PARTICLES * DIM * sizeof(float));
    hipMalloc(&d_vel, NUM_PARTICLES * DIM * sizeof(float));
    hipMalloc(&d_acc, NUM_PARTICLES * DIM * sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_pos, h_pos, NUM_PARTICLES * DIM * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_vel, h_vel, NUM_PARTICLES * DIM * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_acc, h_acc, NUM_PARTICLES * DIM * sizeof(float), hipMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 32;
    int gridSize = (NUM_PARTICLES + blockSize - 1) / blockSize;

    // Kernel launch
    update_positions<<<gridSize, blockSize>>>(d_pos, d_vel, d_acc, NUM_PARTICLES, DT);
    hipDeviceSynchronize();
    calculate_forces<<<gridSize, blockSize, blockSize * DIM * sizeof(float)>>>(d_pos, d_vel, d_acc, NUM_PARTICLES, DT);
    hipDeviceSynchronize();

    // Copy results back to host
    hipMemcpy(h_pos, d_pos, NUM_PARTICLES * DIM * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_vel, d_vel, NUM_PARTICLES * DIM * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_acc, d_acc, NUM_PARTICLES * DIM * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_pos);
    hipFree(d_vel);
    hipFree(d_acc);

    // Free host memory
    delete[] h_pos;
    delete[] h_vel;
    delete[] h_acc;

    // Check for any errors launching the kernel
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    return 0;
}