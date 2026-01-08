#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <hiprand/hiprand_kernel.h>

#define WIDTH 1024
#define HEIGHT 1024
#define MAX_ITERATIONS 100

// Kernel to initialize the random image
__global__ void initRandomImage(unsigned char* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        hiprandState state;
        hiprand_init(idx, 0, 0, &state);
        image[idx] = static_cast<unsigned char>(hiprand(&state) % 256);
    }
}

// Region growing kernel using persistent threads
__global__ void regionGrowing(unsigned char* image, unsigned char* labels, int width, int height, int seedX, int seedY, unsigned char threshold) {
    extern __shared__ unsigned char shared_image[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    // Load image data into shared memory
    if (x < width && y < height) {
        shared_image[threadIdx.y * blockDim.x + threadIdx.x] = image[idx];
    }
    __syncthreads();

    // Avoid bank conflicts by using a 2D shared memory layout
    int sharedIdx = threadIdx.y * (blockDim.x + 1) + threadIdx.x;

    // Persistent threads design pattern
    int iteration = 0;
    bool active = (x == seedX && y == seedY);
    while (active && iteration < MAX_ITERATIONS) {
        if (x < width && y < height) {
            unsigned char center = shared_image[threadIdx.y * (blockDim.x + 1) + threadIdx.x];
            unsigned char left = (threadIdx.x > 0) ? shared_image[threadIdx.y * (blockDim.x + 1) + threadIdx.x - 1] : 0;
            unsigned char right = (threadIdx.x < blockDim.x - 1) ? shared_image[threadIdx.y * (blockDim.x + 1) + threadIdx.x + 1] : 0;
            unsigned char top = (threadIdx.y > 0) ? shared_image[(threadIdx.y - 1) * (blockDim.x + 1) + threadIdx.x] : 0;
            unsigned char bottom = (threadIdx.y < blockDim.y - 1) ? shared_image[(threadIdx.y + 1) * (blockDim.x + 1) + threadIdx.x] : 0;

            if (abs(left - center) <= threshold || abs(right - center) <= threshold || abs(top - center) <= threshold || abs(bottom - center) <= threshold) {
                labels[idx] = 1;
            } else {
                active = false;
            }
        }
        __syncthreads();
        iteration++;
    }
}

int main() {
    unsigned char* d_image;
    unsigned char* d_labels;
    unsigned char* h_labels = new unsigned char[WIDTH * HEIGHT];
    thrust::host_vector<unsigned char> h_image(WIDTH * HEIGHT);

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_image, WIDTH * HEIGHT * sizeof(unsigned char));
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_labels, WIDTH * HEIGHT * sizeof(unsigned char));
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize random image
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);
    hiprandState* devStates;
    hipMalloc((void**)&devStates, WIDTH * HEIGHT * sizeof(hiprandState));
    initRandomImage<<<gridSize, blockSize>>>(d_image, WIDTH, HEIGHT);
    hipDeviceSynchronize();

    // Set seed point and threshold
    int seedX = WIDTH / 2;
    int seedY = HEIGHT / 2;
    unsigned char threshold = 20;

    // Region growing
    size_t sharedMemSize = (blockSize.x + 1) * (blockSize.y + 1) * sizeof(unsigned char);
    regionGrowing<<<gridSize, blockSize, sharedMemSize>>>(d_image, d_labels, WIDTH, HEIGHT, seedX, seedY, threshold);
    hipDeviceSynchronize();

    // Copy results back to host
    err = hipMemcpy(h_labels, d_labels, WIDTH * HEIGHT * sizeof(unsigned char), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_image);
    hipFree(d_labels);

    // Output result (simplified: just print a small portion)
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            std::cout << (int)h_labels[i * WIDTH + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] h_labels;
    return 0;
}