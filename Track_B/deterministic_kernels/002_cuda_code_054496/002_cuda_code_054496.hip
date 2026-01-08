// raytracer.cu
// A simple raytracer using persistent threads optimized for NVIDIA Ampere architecture

#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define a simple 3D vector structure
struct vec3 {
    float x, y, z;
};

// Define a simple sphere object
struct Sphere {
    vec3 center;
    float radius;
};

// Define a ray structure
struct Ray {
    vec3 origin;
    vec3 direction;
};

// Number of objects in the scene
const int NUM_OBJECTS = 1024;

// Kernel to initialize random spheres
__global__ void initSpheres(Sphere *spheres) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_OBJECTS) {
        hiprandState state;
        hiprand_init(idx, 0, 0, &state);
        spheres[idx].center = {hiprand_uniform(&state) * 20.0f - 10.0f,
                               hiprand_uniform(&state) * 20.0f - 10.0f,
                               hiprand_uniform(&state) * 20.0f - 10.0f};
        spheres[idx].radius = hiprand_uniform(&state) * 2.0f + 1.0f;
    }
}

// Kernel to perform ray-sphere intersection
__global__ void raytrace(Ray *rays, Sphere *spheres, int *hits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_OBJECTS) {
        Ray ray = rays[idx];
        Sphere sphere = spheres[idx];
        vec3 oc = {ray.origin.x - sphere.center.x,
                  ray.origin.y - sphere.center.y,
                  ray.origin.z - sphere.center.z};
        float a = ray.direction.x * ray.direction.x +
                  ray.direction.y * ray.direction.y +
                  ray.direction.z * ray.direction.z;
        float b = 2.0f * (oc.x * ray.direction.x +
                         oc.y * ray.direction.y +
                         oc.z * ray.direction.z);
        float c = oc.x * oc.x +
                  oc.y * oc.y +
                  oc.z * oc.z -
                  sphere.radius * sphere.radius;
        float discriminant = b * b - 4.0f * a * c;
        hits[idx] = (discriminant > 0.0f);
    }
}

int main() {
    // Allocate memory for spheres and rays
    Sphere *d_spheres;
    Ray *d_rays;
    int *d_hits;
    hipMalloc(&d_spheres, NUM_OBJECTS * sizeof(Sphere));
    hipMalloc(&d_rays, NUM_OBJECTS * sizeof(Ray));
    hipMalloc(&d_hits, NUM_OBJECTS * sizeof(int));

    // Initialize random spheres
    int blockSize = 256;
    int numBlocks = (NUM_OBJECTS + blockSize - 1) / blockSize;
    initSpheres<<<numBlocks, blockSize>>>(d_spheres);
    hipDeviceSynchronize();
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing spheres: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize rays (simple setup for demonstration)
    Ray *h_rays = new Ray[NUM_OBJECTS];
    for (int i = 0; i < NUM_OBJECTS; ++i) {
        h_rays[i].origin = {0.0f, 0.0f, 0.0f};
        h_rays[i].direction = {1.0f, 0.0f, 0.0f};
    }
    hipMemcpy(d_rays, h_rays, NUM_OBJECTS * sizeof(Ray), hipMemcpyHostToDevice);

    // Perform ray tracing
    raytrace<<<numBlocks, blockSize>>>(d_rays, d_spheres, d_hits);
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error in ray tracing: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    int *h_hits = new int[NUM_OBJECTS];
    hipMemcpy(h_hits, d_hits, NUM_OBJECTS * sizeof(int), hipMemcpyDeviceToHost);

    // Output results
    int hit_count = 0;
    for (int i = 0; i < NUM_OBJECTS; ++i) {
        if (h_hits[i]) {
            hit_count++;
        }
    }
    std::cout << "Number of hits: " << hit_count << std::endl;

    // Clean up
    delete[] h_rays;
    delete[] h_hits;
    hipFree(d_spheres);
    hipFree(d_rays);
    hipFree(d_hits);

    return 0;
}