#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

namespace cg = cooperative_groups;

// Define a simple struct for the priority queue element
struct PQElement {
    int key;
    int value;
};

// Comparator for sorting elements in descending order based on key
__device__ bool operator<(const PQElement& a, const PQElement& b) {
    return a.key > b.key;
}

// Kernel to build a max heap using cooperative groups
__global__ void buildMaxHeap(PQElement* d_elements, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    // Ensure all threads have loaded their data
    cg::sync(cta);

    // Perform the heapify operation in a bottom-up manner
    for (int i = (n / 2) - 1; i >= 0; i--) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (left < n && d_elements[left].key > d_elements[largest].key) {
            largest = left;
        }
        if (right < n && d_elements[right].key > d_elements[largest].key) {
            largest = right;
        }

        if (largest != i) {
            PQElement temp = d_elements[i];
            d_elements[i] = d_elements[largest];
            d_elements[largest] = temp;
        }
    }

    // Ensure all threads have finished heapifying
    cg::sync(cta);
}

// Kernel to extract the maximum element from the heap
__global__ void extractMax(PQElement* d_elements, int n) {
    cg::thread_block cta = cg::this_thread_block();

    // Only the first thread extracts the maximum element
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Swap the first and last element
        PQElement temp = d_elements[0];
        d_elements[0] = d_elements[n - 1];
        d_elements[n - 1] = temp;

        // Heapify the root element
        int largest = 0;
        while (true) {
            int left = 2 * largest + 1;
            int right = 2 * largest + 2;
            int newLargest = largest;

            if (left < n - 1 && d_elements[left].key > d_elements[newLargest].key) {
                newLargest = left;
            }
            if (right < n - 1 && d_elements[right].key > d_elements[newLargest].key) {
                newLargest = right;
            }

            if (newLargest == largest) {
                break;
            }

            // Swap the largest child with the current root
            PQElement temp = d_elements[largest];
            d_elements[largest] = d_elements[newLargest];
            d_elements[newLargest] = temp;

            largest = newLargest;
        }
    }

    // Ensure all threads have finished extracting the max
    cg::sync(cta);
}

int main() {
    const int numElements = 1024;
    thrust::host_vector<PQElement> h_elements(numElements);

    // Initialize elements with random keys and values
    for (int i = 0; i < numElements; ++i) {
        h_elements[i].key = rand() % 1000;
        h_elements[i].value = i;
    }

    thrust::device_vector<PQElement> d_elements = h_elements;

    // Launch kernel to build max heap
    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    buildMaxHeap<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_elements.data()), numElements);
    hipDeviceSynchronize();

    // Check for errors in kernel execution
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error in buildMaxHeap: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Extract the maximum element
    extractMax<<<1, 1>>>(thrust::raw_pointer_cast(d_elements.data()), numElements);
    hipDeviceSynchronize();

    // Check for errors in kernel execution
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error in extractMax: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy result back to host
    thrust::host_vector<PQElement> h_result = d_elements;

    // Print the maximum element
    std::cout << "Maximum element key: " << h_result[numElements - 1].key
              << ", value: " << h_result[numElements - 1].value << std::endl;

    return 0;
}