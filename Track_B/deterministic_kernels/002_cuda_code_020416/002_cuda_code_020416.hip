#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define the image size and other constants
#define WIDTH 512
#define HEIGHT 512
#define BLOCK_SIZE 16
#define MAX_ITERATIONS 100

// Kernel to initialize the image with random values
__global__ void initImage(unsigned char* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        hiprandState state;
        hiprand_init(idx, 0, 0, &state);
        image[idx] = static_cast<unsigned char>(hiprand(&state) % 256);
    }
}

// Kernel to perform active contour segmentation
__global__ void activeContour(unsigned char* image, float* contour, int width, int height, float lambda, float mu) {
    extern __shared__ float sharedContour[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    // Load data into shared memory
    if (x < width && y < height) {
        sharedContour[threadIdx.y * blockDim.x + threadIdx.x] = contour[idx];
        __syncthreads();

        // Calculate new contour position
        if (threadIdx.x > 0 && threadIdx.x < blockDim.x - 1 && threadIdx.y > 0 && threadIdx.y < blockDim.y - 1) {
            float c = sharedContour[threadIdx.y * blockDim.x + threadIdx.x];
            float cN = sharedContour[(threadIdx.y - 1) * blockDim.x + threadIdx.x];
            float cS = sharedContour[(threadIdx.y + 1) * blockDim.x + threadIdx.x];
            float cE = sharedContour[threadIdx.y * blockDim.x + threadIdx.x + 1];
            float cW = sharedContour[threadIdx.y * blockDim.x + threadIdx.x - 1];

            float cNE = sharedContour[(threadIdx.y - 1) * blockDim.x + threadIdx.x + 1];
            float cNW = sharedContour[(threadIdx.y - 1) * blockDim.x + threadIdx.x - 1];
            float cSE = sharedContour[(threadIdx.y + 1) * blockDim.x + threadIdx.x + 1];
            float cSW = sharedContour[(threadIdx.y + 1) * blockDim.x + threadIdx.x - 1];

            float intEnergy = lambda * (cN + cS + cE + cW - 4 * c);
            float extEnergy = mu * (image[idx] - 128) * (image[idx] - 128);
            contour[idx] = c + intEnergy - extEnergy;
        }
        __syncthreads();
    }
}

int main() {
    unsigned char* d_image;
    float* d_contour;
    unsigned char* h_image = new unsigned char[WIDTH * HEIGHT];
    float* h_contour = new float[WIDTH * HEIGHT];

    // Initialize host contour
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        h_contour[i] = 0.0f;
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_image, WIDTH * HEIGHT * sizeof(unsigned char));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_image: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_contour, WIDTH * HEIGHT * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_contour: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize random number generator
    hiprandState* d_state;
    err = hipMalloc((void**)&d_state, WIDTH * HEIGHT * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_state: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize image on device
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);
    initImage<<<gridSize, blockSize>>>(d_image, WIDTH, HEIGHT);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error in initImage kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy contour to device
    err = hipMemcpy(d_contour, h_contour, WIDTH * HEIGHT * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying h_contour to d_contour: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Perform active contour segmentation
    float lambda = 0.1f;
    float mu = 0.1f;
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        activeContour<<<gridSize, blockSize, BLOCK_SIZE * BLOCK_SIZE * sizeof(float)>>>(d_image, d_contour, WIDTH, HEIGHT, lambda, mu);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in activeContour kernel: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy contour back to host
    err = hipMemcpy(h_contour, d_contour, WIDTH * HEIGHT * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying d_contour to h_contour: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_image);
    hipFree(d_contour);
    hipFree(d_state);

    // Free host memory
    delete[] h_image;
    delete[] h_contour;

    return 0;
}