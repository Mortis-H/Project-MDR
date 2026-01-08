#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

using namespace cooperative_groups;

// Define constants
#define N 2048
#define DT 0.01f
#define H 0.1f
#define EPSILON 0.001f
#define G 9.81f
#define MAX_ITERATIONS 10

// Particle structure
struct Particle {
    float3 pos;
    float3 vel;
    float3 acc;
    float mass;
    float rho;
    float p;
};

// Kernel to compute density and pressure
__global__ void computeDensityAndPressure(Particle* particles, float3* d_force, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Particle pi = particles[idx];
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float sum_density = 0.0f;

    for (int j = 0; j < n; j++) {
        if (idx == j) continue;

        Particle pj = particles[j];
        float3 r = make_float3(pi.pos.x - pj.pos.x, pi.pos.y - pj.pos.y, pi.pos.z - pj.pos.z);
        float r_len = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

        if (r_len < H) {
            float q = r_len / H;
            float w = 315.0f / (64.0f * M_PI * powf(H, 9)) * powf(H - q, 3);

            sum_density += pj.mass * w;
        }
    }

    float k = 1000.0f; // Stiffness coefficient
    float rest_density = 1000.0f; // Rest density
    pi.rho = sum_density;
    pi.p = k * (pi.rho - rest_density);

    particles[idx] = pi;
    d_force[idx] = force;
}

// Kernel to compute forces
__global__ void computeForces(Particle* particles, float3* d_force, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Particle pi = particles[idx];
    float3 force = make_float3(0.0f, 0.0f, 0.0f);

    for (int j = 0; j < n; j++) {
        if (idx == j) continue;

        Particle pj = particles[j];
        float3 r = make_float3(pi.pos.x - pj.pos.x, pi.pos.y - pj.pos.y, pi.pos.z - pj.pos.z);
        float r_len = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

        if (r_len < H) {
            float q = r_len / H;
            float w_poly6 = 315.0f / (64.0f * M_PI * powf(H, 9)) * powf(H - q, 3);
            float w_spiky = -45.0f / (M_PI * powf(H, 6)) * powf(H - q, 2);

            float3 grad_w_spiky = make_float3(
                w_spiky * r.x / (r_len + EPSILON),
                w_spiky * r.y / (r_len + EPSILON),
                w_spiky * r.z / (r_len + EPSILON)
            );

            float3 vij = make_float3(pi.vel.x - pj.vel.x, pi.vel.y - pj.vel.y, pi.vel.z - pj.vel.z);
            float3 lap_w_vis = make_float3(0.0f, 0.0f, 0.0f); // Placeholder for viscosity term

            force.x += -pi.mass * pj.mass / pj.rho * (pi.p / (pi.rho * pi.rho) + pj.p / (pj.rho * pj.rho)) * grad_w_spiky.x + pi.mass * vij.x * lap_w_vis.x;
            force.y += -pi.mass * pj.mass / pj.rho * (pi.p / (pi.rho * pi.rho) + pj.p / (pj.rho * pj.rho)) * grad_w_spiky.y + pi.mass * vij.y * lap_w_vis.y;
            force.z += -pi.mass * pj.mass / pj.rho * (pi.p / (pi.rho * pi.rho) + pj.p / (pj.rho * pj.rho)) * grad_w_spiky.z + pi.mass * vij.z * lap_w_vis.z;
        }
    }

    force.y -= pi.mass * G; // Gravity
    d_force[idx] = force;
}

// Kernel to update positions and velocities
__global__ void updateParticles(Particle* particles, float3* d_force, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Particle pi = particles[idx];
    float3 force = d_force[idx];

    pi.acc = make_float3(force.x / pi.mass, force.y / pi.mass, force.z / pi.mass);
    pi.vel.x += pi.acc.x * DT;
    pi.vel.y += pi.acc.y * DT;
    pi.vel.z += pi.acc.z * DT;
    pi.pos.x += pi.vel.x * DT;
    pi.pos.y += pi.vel.y * DT;
    pi.pos.z += pi.vel.z * DT;

    particles[idx] = pi;
}

// Error checking macro
#define cudaCheckError() {                                          \
    hipError_t e=hipGetLastError();                               \
    if(e!=hipSuccess) {                                            \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

int main() {
    // Initialize particles
    Particle* h_particles = new Particle[N];
    for (int i = 0; i < N; i++) {
        h_particles[i].pos = make_float3(i * 0.1f, 0.0f, 0.0f);
        h_particles[i].vel = make_float3(0.0f, 0.0f, 0.0f);
        h_particles[i].acc = make_float3(0.0f, 0.0f, 0.0f);
        h_particles[i].mass = 1.0f;
        h_particles[i].rho = 0.0f;
        h_particles[i].p = 0.0f;
    }

    // Allocate device memory
    Particle* d_particles;
    float3* d_force;
    hipMalloc((void**)&d_particles, N * sizeof(Particle));
    hipMalloc((void**)&d_force, N * sizeof(float3));
    cudaCheckError();

    // Copy particles to device
    hipMemcpy(d_particles, h_particles, N * sizeof(Particle), hipMemcpyHostToDevice);
    cudaCheckError();

    // Launch kernels
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        computeDensityAndPressure<<<numBlocks, blockSize>>>(d_particles, d_force, N);
        cudaCheckError();
        computeForces<<<numBlocks, blockSize>>>(d_particles, d_force, N);
        cudaCheckError();
        updateParticles<<<numBlocks, blockSize>>>(d_particles, d_force, N);
        cudaCheckError();
    }

    // Copy particles back to host
    hipMemcpy(h_particles, d_particles, N * sizeof(Particle), hipMemcpyDeviceToHost);
    cudaCheckError();

    // Free device memory
    hipFree(d_particles);
    hipFree(d_force);
    cudaCheckError();

    // Clean up
    delete[] h_particles;

    return 0;
}