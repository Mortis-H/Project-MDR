#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#define N 8192

__global__ void insert_into_heap(int *heap, int value, int *heap_size) {
    int idx = atomicAdd(heap_size, 1);
    heap[idx] = value;
    while (idx > 0) {
        int parent = (idx - 1) / 2;
        if (heap[idx] <= heap[parent]) break;
        int temp = heap[idx];
        heap[idx] = heap[parent];
        heap[parent] = temp;
        idx = parent;
    }
}

__global__ void extract_max_from_heap(int *heap, int *heap_size) {
    int size = atomicSub(heap_size, 1);
    if (size > 0) {
        heap[0] = heap[size - 1];
        int idx = 0;
        while (true) {
            int left_child = 2 * idx + 1;
            int right_child = 2 * idx + 2;
            int largest = idx;
            if (left_child < size - 1 && heap[left_child] > heap[largest]) largest = left_child;
            if (right_child < size - 1 && heap[right_child] > heap[largest]) largest = right_child;
            if (largest == idx) break;
            int temp = heap[idx];
            heap[idx] = heap[largest];
            heap[largest] = temp;
            idx = largest;
        }
    }
}

int main() {
    thrust::host_vector<int> h_heap(N, 0);
    thrust::device_vector<int> d_heap(h_heap);
    thrust::device_vector<int> d_heap_size(1, 0);

    // Insert elements into the heap
    for (int i = 0; i < N; ++i) {
        insert_into_heap<<<1, 1>>>(thrust::raw_pointer_cast(d_heap.data()), i, thrust::raw_pointer_cast(d_heap_size.data()));
        hipDeviceSynchronize();
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Extract elements from the heap
    for (int i = 0; i < N; ++i) {
        extract_max_from_heap<<<1, 1>>>(thrust::raw_pointer_cast(d_heap.data()), thrust::raw_pointer_cast(d_heap_size.data()));
        hipDeviceSynchronize();
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy result back to host
    thrust::copy(d_heap.begin(), d_heap.end(), h_heap.begin());

    // Verify the result (should be sorted in ascending order)
    thrust::sort(h_heap.begin(), h_heap.end(), thrust::greater<int>());
    for (int i = 0; i < N; ++i) {
        if (h_heap[i] != i) {
            std::cerr << "Heap sort failed at index " << i << std::endl;
            return -1;
        }
    }

    std::cout << "Heap operations completed successfully!" << std::endl;
    return 0;
}