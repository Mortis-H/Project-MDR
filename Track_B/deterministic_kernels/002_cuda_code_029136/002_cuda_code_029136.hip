#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_NODES 512
#define MAX_COLOR 8

// Kernel to perform graph coloring
__global__ void graphColoringKernel(int* graph, int* colors) {
    extern __shared__ int shared_colors[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Load the color for this node into shared memory
    if (idx < NUM_NODES) {
        shared_colors[threadIdx.x] = colors[idx];
        __syncthreads();

        // Try to color the node
        for (int color = 0; color < MAX_COLOR; ++color) {
            bool valid = true;
            // Check all neighbors
            for (int j = 0; j < NUM_NODES; ++j) {
                if (graph[idx * NUM_NODES + j] == 1 && shared_colors[j] == color) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                colors[idx] = color;
                break;
            }
        }
    }
}

// Function to initialize the graph randomly
__global__ void initializeGraphKernel(int* graph) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < NUM_NODES * NUM_NODES) {
        hiprandState state;
        hiprand_init(idx, 0, 0, &state);
        graph[idx] = (hiprand(&state) % 2) && (idx % (NUM_NODES + 1) != 0); // No self-loops
    }
}

int main() {
    int* h_graph;
    int* h_colors;
    int* d_graph;
    int* d_colors;

    // Allocate memory for the graph and colors on the host
    h_graph = (int*)malloc(NUM_NODES * NUM_NODES * sizeof(int));
    h_colors = (int*)malloc(NUM_NODES * sizeof(int));

    // Initialize colors to -1 (uncolored)
    for (int i = 0; i < NUM_NODES; ++i) {
        h_colors[i] = -1;
    }

    // Allocate memory for the graph and colors on the device
    hipMalloc((void**)&d_graph, NUM_NODES * NUM_NODES * sizeof(int));
    hipMalloc((void**)&d_colors, NUM_NODES * sizeof(int));

    // Initialize the graph on the device
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_NODES * NUM_NODES + threadsPerBlock - 1) / threadsPerBlock;
    initializeGraphKernel<<<blocksPerGrid, threadsPerBlock>>>(d_graph);
    hipDeviceSynchronize();

    // Copy colors to device
    hipMemcpy(d_colors, h_colors, NUM_NODES * sizeof(int), hipMemcpyHostToDevice);

    // Launch the graph coloring kernel
    blocksPerGrid = (NUM_NODES + threadsPerBlock - 1) / threadsPerBlock;
    graphColoringKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_graph, d_colors);
    hipDeviceSynchronize();

    // Copy the results back to the host
    hipMemcpy(h_colors, d_colors, NUM_NODES * sizeof(int), hipMemcpyDeviceToHost);

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Print the colors
    std::cout << "Node colors:" << std::endl;
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << "Node " << i << ": Color " << h_colors[i] << std::endl;
    }

    // Free memory
    free(h_graph);
    free(h_colors);
    hipFree(d_graph);
    hipFree(d_colors);

    return 0;
}