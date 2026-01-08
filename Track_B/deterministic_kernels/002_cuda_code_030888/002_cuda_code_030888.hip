#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cmath>

#define N 128
#define BLOCK_SIZE 32
#define GRID_SIZE (N + BLOCK_SIZE - 1) / BLOCK_SIZE

// Kernel to compute forces using particle mesh
__global__ void computeForces(float4 *pos, float4 *forces, float4 *mesh, int nx, int ny, int nz, float dx, float dy, float dz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float4 p = pos[idx];
    float4 f = {0.0f, 0.0f, 0.0f, 0.0f};

    // Compute grid cell indices
    int ix = static_cast<int>(p.x / dx);
    int iy = static_cast<int>(p.y / dy);
    int iz = static_cast<int>(p.z / dz);

    // Interpolate forces from the mesh
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            for (int k = -1; k <= 1; ++k) {
                int nx_i = (ix + i + nx) % nx;
                int ny_j = (iy + j + ny) % ny;
                int nz_k = (iz + k + nz) % nz;
                float4 meshForce = mesh[nz_k * nx * ny + ny_j * nx + nx_i];
                float wx = 1.0f - std::abs(p.x - (nx_i + 0.5f) * dx) / dx;
                float wy = 1.0f - std::abs(p.y - (ny_j + 0.5f) * dy) / dy;
                float wz = 1.0f - std::abs(p.z - (nz_k + 0.5f) * dz) / dz;
                float weight = wx * wy * wz;
                f.x += weight * meshForce.x;
                f.y += weight * meshForce.y;
                f.z += weight * meshForce.z;
            }
        }
    }

    forces[idx] = f;
}

// Kernel to update positions based on forces
__global__ void updatePositions(float4 *pos, float4 *forces, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float4 p = pos[idx];
    float4 f = forces[idx];
    p.x += f.x * dt;
    p.y += f.y * dt;
    p.z += f.z * dt;
    pos[idx] = p;
}

int main() {
    // Parameters
    int nx = 16, ny = 16, nz = 16;
    float dx = 1.0f / nx, dy = 1.0f / ny, dz = 1.0f / nz;
    float dt = 0.01f;

    // Allocate host memory
    std::vector<float4> h_pos(N);
    std::vector<float4> h_forces(N);
    std::vector<float4> h_mesh(nx * ny * nz);

    // Initialize positions and mesh
    for (int i = 0; i < N; ++i) {
        h_pos[i] = {static_cast<float>(i % nx) * dx + 0.5f * dx,
                    static_cast<float>((i / nx) % ny) * dy + 0.5f * dy,
                    static_cast<float>(i / (nx * ny)) * dz + 0.5f * dz,
                    0.0f};
    }
    for (int i = 0; i < nx * ny * nz; ++i) {
        h_mesh[i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    // Allocate device memory
    float4 *d_pos, *d_forces, *d_mesh;
    hipMalloc(&d_pos, N * sizeof(float4));
    hipMalloc(&d_forces, N * sizeof(float4));
    hipMalloc(&d_mesh, nx * ny * nz * sizeof(float4));

    // Copy data to device
    hipMemcpy(d_pos, h_pos.data(), N * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_mesh, h_mesh.data(), nx * ny * nz * sizeof(float4), hipMemcpyHostToDevice);

    // Launch kernels
    computeForces<<<GRID_SIZE, BLOCK_SIZE>>>(d_pos, d_forces, d_mesh, nx, ny, nz, dx, dy, dz);
    hipDeviceSynchronize();
    updatePositions<<<GRID_SIZE, BLOCK_SIZE>>>(d_pos, d_forces, dt);
    hipDeviceSynchronize();

    // Copy results back to host
    hipMemcpy(h_pos.data(), d_pos, N * sizeof(float4), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_pos);
    hipFree(d_forces);
    hipFree(d_mesh);

    // Output first particle's position
    std::cout << "Final position of particle 0: (" << h_pos[0].x << ", " << h_pos[0].y << ", " << h_pos[0].z << ")" << std::endl;

    return 0;
}