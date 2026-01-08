// cloth_simulation.cu
// A research-grade CUDA kernel for cloth simulation optimized for occupancy on Hopper architecture.

#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define grid and block dimensions
#define GRID_WIDTH 128
#define GRID_HEIGHT 128
#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

// Define the number of threads per block
#define THREADS_PER_BLOCK (BLOCK_WIDTH * BLOCK_HEIGHT)

// Define the number of blocks in the grid
#define BLOCKS_X (GRID_WIDTH / BLOCK_WIDTH)
#define BLOCKS_Y (GRID_HEIGHT / BLOCK_HEIGHT)

// Define the number of vertices in the cloth grid
#define NUM_VERTICES (GRID_WIDTH * GRID_HEIGHT)

// Structure to represent a vertex in the cloth
struct Vertex {
    float x, y, z;  // Position
    float vx, vy, vz; // Velocity
    float fx, fy, fz; // Force
};

// Kernel to update the positions and velocities of vertices
__global__ void updateClothVertices(Vertex* vertices, float deltaTime, float damping) {
    // Calculate the global thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * GRID_WIDTH + x;

    // Ensure the thread index is within the grid bounds
    if (x < GRID_WIDTH && y < GRID_HEIGHT) {
        // Calculate new velocity and position
        vertices[index].vx += vertices[index].fx * deltaTime;
        vertices[index].vy += vertices[index].fy * deltaTime;
        vertices[index].vz += vertices[index].fz * deltaTime;

        // Apply damping
        vertices[index].vx *= damping;
        vertices[index].vy *= damping;
        vertices[index].vz *= damping;

        // Update position
        vertices[index].x += vertices[index].vx * deltaTime;
        vertices[index].y += vertices[index].vy * deltaTime;
        vertices[index].z += vertices[index].vz * deltaTime;

        // Reset forces for the next iteration
        vertices[index].fx = 0.0f;
        vertices[index].fy = 0.0f;
        vertices[index].fz = 0.0f;
    }
}

// Kernel to apply forces between connected vertices (simple spring force)
__global__ void applySpringForces(Vertex* vertices, float stiffness, float restLength) {
    // Calculate the global thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * GRID_WIDTH + x;

    // Ensure the thread index is within the grid bounds
    if (x < GRID_WIDTH && y < GRID_HEIGHT) {
        // Apply force from the vertex to the right
        if (x < GRID_WIDTH - 1) {
            int rightIndex = index + 1;
            float dx = vertices[rightIndex].x - vertices[index].x;
            float dy = vertices[rightIndex].y - vertices[index].y;
            float dz = vertices[rightIndex].z - vertices[index].z;
            float distance = sqrtf(dx * dx + dy * dy + dz * dz);
            float force = stiffness * (distance - restLength);
            float fx = force * dx / distance;
            float fy = force * dy / distance;
            float fz = force * dz / distance;

            // Apply force to both vertices
            atomicAdd(&vertices[index].fx, -fx);
            atomicAdd(&vertices[index].fy, -fy);
            atomicAdd(&vertices[index].fz, -fz);
            atomicAdd(&vertices[rightIndex].fx, fx);
            atomicAdd(&vertices[rightIndex].fy, fy);
            atomicAdd(&vertices[rightIndex].fz, fz);
        }

        // Apply force from the vertex below
        if (y < GRID_HEIGHT - 1) {
            int downIndex = index + GRID_WIDTH;
            float dx = vertices[downIndex].x - vertices[index].x;
            float dy = vertices[downIndex].y - vertices[index].y;
            float dz = vertices[downIndex].z - vertices[index].z;
            float distance = sqrtf(dx * dx + dy * dy + dz * dz);
            float force = stiffness * (distance - restLength);
            float fx = force * dx / distance;
            float fy = force * dy / distance;
            float fz = force * dz / distance;

            // Apply force to both vertices
            atomicAdd(&vertices[index].fx, -fx);
            atomicAdd(&vertices[index].fy, -fy);
            atomicAdd(&vertices[index].fz, -fz);
            atomicAdd(&vertices[downIndex].fx, fx);
            atomicAdd(&vertices[downIndex].fy, fy);
            atomicAdd(&vertices[downIndex].fz, fz);
        }
    }
}

int main() {
    // Define simulation parameters
    float deltaTime = 0.016f; // Time step
    float damping = 0.99f;    // Damping factor
    float stiffness = 100.0f; // Stiffness of springs
    float restLength = 1.0f;  // Rest length of springs

    // Allocate memory for vertices on the host
    Vertex* h_vertices = new Vertex[NUM_VERTICES];

    // Initialize vertices (simple grid initialization)
    for (int y = 0; y < GRID_HEIGHT; ++y) {
        for (int x = 0; x < GRID_WIDTH; ++x) {
            int index = y * GRID_WIDTH + x;
            h_vertices[index].x = static_cast<float>(x);
            h_vertices[index].y = static_cast<float>(y);
            h_vertices[index].z = 0.0f;
            h_vertices[index].vx = 0.0f;
            h_vertices[index].vy = 0.0f;
            h_vertices[index].vz = 0.0f;
            h_vertices[index].fx = 0.0f;
            h_vertices[index].fy = 0.0f;
            h_vertices[index].fz = 0.0f;
        }
    }

    // Allocate memory for vertices on the device
    Vertex* d_vertices;
    hipError_t err = hipMalloc(&d_vertices, NUM_VERTICES * sizeof(Vertex));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory: " << hipGetErrorString(err) << std::endl;
        delete[] h_vertices;
        return -1;
    }

    // Copy vertices from host to device
    err = hipMemcpy(d_vertices, h_vertices, NUM_VERTICES * sizeof(Vertex), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_vertices);
        delete[] h_vertices;
        return -1;
    }

    // Define grid and block dimensions
    dim3 blockSize(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 gridSize(BLOCKS_X, BLOCKS_Y);

    // Simulate cloth for a number of steps
    for (int step = 0; step < 100; ++step) {
        // Apply spring forces
        applySpringForces<<<gridSize, blockSize>>>(d_vertices, stiffness, restLength);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Kernel launch failed (applySpringForces): " << hipGetErrorString(err) << std::endl;
            hipFree(d_vertices);
            delete[] h_vertices;
            return -1;
        }

        // Update cloth vertices
        updateClothVertices<<<gridSize, blockSize>>>(d_vertices, deltaTime, damping);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Kernel launch failed (updateClothVertices): " << hipGetErrorString(err) << std::endl;
            hipFree(d_vertices);
            delete[] h_vertices;
            return -1;
        }

        // Synchronize to ensure all operations are complete
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::cerr << "Failed to synchronize device: " << hipGetErrorString(err) << std::endl;
            hipFree(d_vertices);
            delete[] h_vertices;
            return -1;
        }
    }

    // Copy vertices back to host
    err = hipMemcpy(h_vertices, d_vertices, NUM_VERTICES * sizeof(Vertex), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_vertices);
        delete[] h_vertices;
        return -1;
    }

    // Free device memory
    hipFree(d_vertices);

    // Free host memory
    delete[] h_vertices;

    // Output a success message
    std::cout << "Cloth simulation completed successfully." << std::endl;

    return 0;
}