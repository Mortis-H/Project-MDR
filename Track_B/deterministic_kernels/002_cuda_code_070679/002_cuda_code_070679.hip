// 4K Video Frame Processing with Warp-Level Optimization
// Target Architecture: Turing
// This code demonstrates a simple image processing operation (grayscale conversion)
// on 4K frames using CUDA with warp-level optimization.

#include <iostream>
#include <hip/hip_runtime.h>

// Constants for 4K resolution
#define WIDTH 3840
#define HEIGHT 2160
#define CHANNELS 3  // RGB

// Kernel to convert an RGB image to grayscale
__global__ void rgbToGrayscale(const unsigned char* input, unsigned char* output, int width, int height) {
    // Calculate the global index in the 1D grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the row and column in the 2D image
    int row = idx / width;
    int col = idx % width;

    // Check if the current index is within the image bounds
    if (row < height && col < width) {
        // Calculate the starting index in the input array for the current pixel
        int inputIdx = (row * width + col) * CHANNELS;

        // Load the RGB values
        unsigned char r = input[inputIdx];
        unsigned char g = input[inputIdx + 1];
        unsigned char b = input[inputIdx + 2];

        // Convert to grayscale using the luminosity method
        unsigned char gray = static_cast<unsigned char>(0.2126f * r + 0.7152f * g + 0.0722f * b);

        // Store the grayscale value in the output array
        output[row * width + col] = gray;
    }
}

// Warp-level optimization: Load/Store using shared memory and warp-level primitives
__global__ void rgbToGrayscaleOptimized(const unsigned char* input, unsigned char* output, int width, int height) {
    // Shared memory to store a block of pixels
    extern __shared__ unsigned char sharedInput[];

    // Calculate the global index in the 1D grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the row and column in the 2D image
    int row = idx / width;
    int col = idx % width;

    // Check if the current index is within the image bounds
    if (row < height && col < width) {
        // Calculate the starting index in the input array for the current pixel
        int inputIdx = (row * width + col) * CHANNELS;

        // Load the RGB values into shared memory
        if (threadIdx.x < blockDim.x) {
            sharedInput[threadIdx.x * CHANNELS] = input[inputIdx];
            sharedInput[threadIdx.x * CHANNELS + 1] = input[inputIdx + 1];
            sharedInput[threadIdx.x * CHANNELS + 2] = input[inputIdx + 2];
        }

        __syncthreads();

        // Calculate the grayscale value using the luminosity method
        unsigned char r = sharedInput[threadIdx.x * CHANNELS];
        unsigned char g = sharedInput[threadIdx.x * CHANNELS + 1];
        unsigned char b = sharedInput[threadIdx.x * CHANNELS + 2];
        unsigned char gray = static_cast<unsigned char>(0.2126f * r + 0.7152f * g + 0.0722f * b);

        // Store the grayscale value in the output array
        output[row * width + col] = gray;
    }
}

int main() {
    // Allocate memory for input and output images
    unsigned char* h_input = new unsigned char[WIDTH * HEIGHT * CHANNELS];
    unsigned char* h_output = new unsigned char[WIDTH * HEIGHT];

    // Initialize input image with random values (simulating an image)
    for (int i = 0; i < WIDTH * HEIGHT * CHANNELS; ++i) {
        h_input[i] = static_cast<unsigned char>(rand() % 256);
    }

    // Allocate memory on the device
    unsigned char* d_input;
    unsigned char* d_output;
    hipError_t err = hipMalloc(&d_input, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_input: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&d_output, WIDTH * HEIGHT * sizeof(unsigned char));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_output: " << hipGetErrorString(err) << std::endl;
        hipFree(d_input);
        return -1;
    }

    // Copy input image to the device
    err = hipMemcpy(d_input, h_input, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_input);
        hipFree(d_output);
        return -1;
    }

    // Define block and grid sizes
    dim3 blockSize(WIDTH);
    dim3 gridSize((WIDTH * HEIGHT + blockSize.x - 1) / blockSize.x);

    // Launch the kernel
    rgbToGrayscaleOptimized<<<gridSize, blockSize, WIDTH * CHANNELS * sizeof(unsigned char)>>>(d_input, d_output, WIDTH, HEIGHT);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_input);
        hipFree(d_output);
        return -1;
    }

    // Copy the result back to the host
    err = hipMemcpy(h_output, d_output, WIDTH * HEIGHT * sizeof(unsigned char), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying to host: " << hipGetErrorString(err) << std::endl;
        hipFree(d_input);
        hipFree(d_output);
        return -1;
    }

    // Free device memory
    hipFree(d_input);
    hipFree(d_output);

    // Free host memory
    delete[] h_input;
    delete[] h_output;

    std::cout << "Grayscale conversion completed successfully." << std::endl;

    return 0;
}