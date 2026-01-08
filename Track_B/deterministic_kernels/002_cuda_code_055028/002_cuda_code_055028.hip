#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#define PQ_SIZE 32

__global__ void insert_into_pq(int *pq, int *new_element, int *size) {
    int idx = threadIdx.x;
    if (idx >= *size) return;

    // Shift elements to the right to make space for the new element
    for (int i = *size; i > idx && pq[i - 1] < *new_element; --i) {
        pq[i] = pq[i - 1];
    }
    if (pq[idx] < *new_element) {
        pq[idx] = *new_element;
    }

    // Ensure the size is updated atomically
    atomicAdd(size, 1);
}

__global__ void extract_max(int *pq, int *max_element, int *size) {
    int idx = threadIdx.x;
    if (idx != 0) return;

    // Extract the maximum element
    *max_element = pq[0];

    // Shift elements to the left to fill the gap
    for (int i = 0; i < *size - 1; ++i) {
        pq[i] = pq[i + 1];
    }

    // Ensure the size is updated atomically
    atomicSub(size, 1);
}

int main() {
    // Initialize host and device vectors
    thrust::host_vector<int> h_pq(PQ_SIZE, 0);
    thrust::host_vector<int> h_new_element(1, 42);
    thrust::host_vector<int> h_max_element(1, 0);
    thrust::host_vector<int> h_size(1, 0);

    thrust::device_vector<int> d_pq = h_pq;
    thrust::device_vector<int> d_new_element = h_new_element;
    thrust::device_vector<int> d_max_element = h_max_element;
    thrust::device_vector<int> d_size = h_size;

    // Create CUDA streams for asynchronous operations
    hipStream_t stream1, stream2;
    hipStreamCreate(&stream1);
    hipStreamCreate(&stream2);

    // Insert elements into the priority queue asynchronously
    for (int i = 0; i < 10; ++i) {
        h_new_element[0] = i * 5;
        hipMemcpyAsync(thrust::raw_pointer_cast(d_new_element.data()), thrust::raw_pointer_cast(h_new_element.data()), sizeof(int), hipMemcpyHostToDevice, stream1);
        insert_into_pq<<<1, PQ_SIZE, 0, stream1>>>(thrust::raw_pointer_cast(d_pq.data()), thrust::raw_pointer_cast(d_new_element.data()), thrust::raw_pointer_cast(d_size.data()));
        hipStreamSynchronize(stream1);
    }

    // Extract elements from the priority queue asynchronously
    for (int i = 0; i < 5; ++i) {
        extract_max<<<1, 1, 0, stream2>>>(thrust::raw_pointer_cast(d_pq.data()), thrust::raw_pointer_cast(d_max_element.data()), thrust::raw_pointer_cast(d_size.data()));
        hipMemcpyAsync(thrust::raw_pointer_cast(h_max_element.data()), thrust::raw_pointer_cast(d_max_element.data()), sizeof(int), hipMemcpyDeviceToHost, stream2);
        hipStreamSynchronize(stream2);
        std::cout << "Extracted max element: " << h_max_element[0] << std::endl;
    }

    // Clean up streams
    hipStreamDestroy(stream1);
    hipStreamDestroy(stream2);

    // Check for any errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    return 0;
}