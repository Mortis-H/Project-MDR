#include <iostream>
#include <hip/hip_runtime.h>

// Define the maximum number of elements in the priority queue
#define MAX_ELEMENTS 16384

// Kernel to insert elements into the priority queue
__global__ void insertIntoPriorityQueue(int* queue, int* elements, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        // Use atomic operations to insert elements into the queue
        atomicAdd(&queue[elements[idx]], 1);
    }
}

// Kernel to extract the maximum element from the priority queue
__global__ void extractMaxFromPriorityQueue(int* queue, int* maxElement) {
    extern __shared__ int sdata[];

    // Load data into shared memory
    int idx = threadIdx.x;
    sdata[idx] = queue[idx];
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (idx < s) {
            sdata[idx] = max(sdata[idx], sdata[idx + s]);
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (idx == 0) {
        atomicMax(maxElement, sdata[0]);
    }
}

int main() {
    // Initialize device
    int device;
    hipGetDevice(&device);
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, device);
    std::cout << "Using device " << device << ": " << deviceProp.name << std::endl;

    // Allocate memory for the priority queue and elements on the host
    int* h_queue = new int[MAX_ELEMENTS];
    int* h_elements = new int[MAX_ELEMENTS];

    // Initialize the priority queue to zero
    for (int i = 0; i < MAX_ELEMENTS; ++i) {
        h_queue[i] = 0;
    }

    // Initialize elements to insert into the priority queue
    for (int i = 0; i < MAX_ELEMENTS; ++i) {
        h_elements[i] = rand() % MAX_ELEMENTS;
    }

    // Allocate memory for the priority queue and elements on the device
    int* d_queue;
    int* d_elements;
    hipMalloc((void**)&d_queue, MAX_ELEMENTS * sizeof(int));
    hipMalloc((void**)&d_elements, MAX_ELEMENTS * sizeof(int));

    // Copy data from host to device
    hipMemcpy(d_queue, h_queue, MAX_ELEMENTS * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_elements, h_elements, MAX_ELEMENTS * sizeof(int), hipMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((MAX_ELEMENTS + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Insert elements into the priority queue
    insertIntoPriorityQueue<<<blocksPerGrid, threadsPerBlock>>>(d_queue, d_elements, MAX_ELEMENTS);
    hipDeviceSynchronize();

    // Check for errors in kernel launch
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error in insertIntoPriorityQueue kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Allocate memory for the maximum element on the host and device
    int h_maxElement = 0;
    int* d_maxElement;
    hipMalloc((void**)&d_maxElement, sizeof(int));
    hipMemcpy(d_maxElement, &h_maxElement, sizeof(int), hipMemcpyHostToDevice);

    // Extract the maximum element from the priority queue
    extractMaxFromPriorityQueue<<<1, threadsPerBlock, threadsPerBlock.x * sizeof(int)>>>(d_queue, d_maxElement);
    hipDeviceSynchronize();

    // Check for errors in kernel launch
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error in extractMaxFromPriorityQueue kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy the maximum element from device to host
    hipMemcpy(&h_maxElement, d_maxElement, sizeof(int), hipMemcpyDeviceToHost);

    // Print the maximum element
    std::cout << "Maximum element in the priority queue: " << h_maxElement << std::endl;

    // Free device memory
    hipFree(d_queue);
    hipFree(d_elements);
    hipFree(d_maxElement);

    // Free host memory
    delete[] h_queue;
    delete[] h_elements;

    return 0;
}