/*
 * This program uses the device CURAND API to calculate what
 * proportion of pseudo-random ints have low bit set.
 * It then generates uniform results to calculate how many
 * are greater than .5.
 * It then generates  normal results to calculate how many
 * are within one standard deviation of the mean.
 */
#include <stdio.h>
#include <stdlib.h>

#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define CUDA_CALL(x) do { if((x) != hipSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(hiprandState *state)
{
    int id = threadIdx.x + blockIdx.x * 64;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    hiprand_init(1234, id, 0, &state[id]);
}

__global__ void setup_kernel(hiprandStatePhilox4_32_10_t *state)
{
    int id = threadIdx.x + blockIdx.x * 64;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    hiprand_init(1234, id, 0, &state[id]);
}

__global__ void setup_kernel(hiprandStateMRG32k3a *state)
{
    int id = threadIdx.x + blockIdx.x * 64;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    hiprand_init(0, id, 0, &state[id]);
}

__global__ void generate_kernel(hiprandState *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * 64;
    int count = 0;
    unsigned int x;
    /* Copy state to local memory for efficiency */
    hiprandState localState = state[id];
    /* Generate pseudo-random unsigned ints */
    for(int i = 0; i < n; i++) {
        x = hiprand(&localState);
        /* Check if low bit set */
        if(x & 1) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_kernel(hiprandStatePhilox4_32_10_t *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * 64;
    int count = 0;
    unsigned int x;
    /* Copy state to local memory for efficiency */
    hiprandStatePhilox4_32_10_t localState = state[id];
    /* Generate pseudo-random unsigned ints */
    for(int i = 0; i < n; i++) {
        x = hiprand(&localState);
        /* Check if low bit set */
        if(x & 1) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_uniform_kernel(hiprandState *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * 64;
    unsigned int count = 0;
    float x;
    /* Copy state to local memory for efficiency */
    hiprandState localState = state[id];
    /* Generate pseudo-random uniforms */
    for(int i = 0; i < n; i++) {
        x = hiprand_uniform(&localState);
        /* Check if > .5 */
        if(x > .5) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_uniform_kernel(hiprandStatePhilox4_32_10_t *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * 64;
    unsigned int count = 0;
    float x;
    /* Copy state to local memory for efficiency */
    hiprandStatePhilox4_32_10_t localState = state[id];
    /* Generate pseudo-random uniforms */
    for(int i = 0; i < n; i++) {
        x = hiprand_uniform(&localState);
        /* Check if > .5 */
        if(x > .5) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_normal_kernel(hiprandState *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * 64;
    unsigned int count = 0;
    float2 x;
    /* Copy state to local memory for efficiency */
    hiprandState localState = state[id];
    /* Generate pseudo-random normals */
    for(int i = 0; i < n/2; i++) {
        x = hiprand_normal2(&localState);
        /* Check if within one standard deviaton */
        if((x.x > -1.0) && (x.x < 1.0)) {
            count++;
        }
        if((x.y > -1.0) && (x.y < 1.0)) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_normal_kernel(hiprandStatePhilox4_32_10_t *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * 64;
    unsigned int count = 0;
    float2 x;
    /* Copy state to local memory for efficiency */
    hiprandStatePhilox4_32_10_t localState = state[id];
    /* Generate pseudo-random normals */
    for(int i = 0; i < n/2; i++) {
        x = hiprand_normal2(&localState);
        /* Check if within one standard deviaton */
        if((x.x > -1.0) && (x.x < 1.0)) {
            count++;
        }
        if((x.y > -1.0) && (x.y < 1.0)) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_kernel(hiprandStateMRG32k3a *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * 64;
    unsigned int count = 0;
    unsigned int x;
    /* Copy state to local memory for efficiency */
    hiprandStateMRG32k3a localState = state[id];
    /* Generate pseudo-random unsigned ints */
    for(int i = 0; i < n; i++) {
        x = hiprand(&localState);
        /* Check if low bit set */
        if(x & 1) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_uniform_kernel(hiprandStateMRG32k3a *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * 64;
    unsigned int count = 0;
    double x;
    /* Copy state to local memory for efficiency */
    hiprandStateMRG32k3a localState = state[id];
    /* Generate pseudo-random uniforms */
    for(int i = 0; i < n; i++) {
        x = hiprand_uniform_double(&localState);
        /* Check if > .5 */
        if(x > .5) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_normal_kernel(hiprandStateMRG32k3a *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * 64;
    unsigned int count = 0;
    double2 x;
    /* Copy state to local memory for efficiency */
    hiprandStateMRG32k3a localState = state[id];
    /* Generate pseudo-random normals */
    for(int i = 0; i < n/2; i++) {
        x = hiprand_normal2_double(&localState);
        /* Check if within one standard deviaton */
        if((x.x > -1.0) && (x.x < 1.0)) {
            count++;
        }
        if((x.y > -1.0) && (x.y < 1.0)) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

int main(int argc, char *argv[])
{

    int i;
    unsigned int total;
    hiprandState *devStates;
    hiprandStateMRG32k3a *devMRGStates;
    hiprandStatePhilox4_32_10_t *devPHILOXStates;
    unsigned int *devResults, *hostResults;
    bool useMRG = 0;
    bool usePHILOX = 0;
    int sampleCount = 10000;
    bool doubleSupported = 0;
    int device;
    struct hipDeviceProp_t properties;

    /* check for double precision support */
    CUDA_CALL(hipGetDevice(&device));
    CUDA_CALL(hipGetDeviceProperties(&properties,device));
    if ( properties.major >= 2 || (properties.major == 1 && properties.minor >= 3) ) {
        doubleSupported = 1;
    }

    /* Check for MRG32k3a option (default is XORWOW) */
    if (argc >= 2)  {
        if (strcmp(argv[1],"-m") == 0) {
            useMRG = 1;
            if (!doubleSupported){
                printf("MRG32k3a requires double precision\n");
                printf("^^^^ test WAIVED due to lack of double precision\n");
                return EXIT_SUCCESS;
            }
        }else if (strcmp(argv[1],"-p") == 0) {
		usePHILOX = 1;
	}
        /* Allow over-ride of sample count */
        sscanf(argv[argc-1],"%d",&sampleCount);
    }

    /* Allocate space for results on host */
    hostResults = (unsigned int *)calloc(64 * 64, sizeof(int));

    /* Allocate space for results on device */
    CUDA_CALL(hipMalloc((void **)&devResults, 64 * 64 *
              sizeof(unsigned int)));

    /* Set results to 0 */
    CUDA_CALL(hipMemset(devResults, 0, 64 * 64 *
              sizeof(unsigned int)));

    /* Allocate space for prng states on device */
    if (useMRG) {
        CUDA_CALL(hipMalloc((void **)&devMRGStates, 64 * 64 *
                  sizeof(hiprandStateMRG32k3a)));
    }else if(usePHILOX) {
        CUDA_CALL(hipMalloc((void **)&devPHILOXStates, 64 * 64 *
                  sizeof(hiprandStatePhilox4_32_10_t)));
    }else {
        CUDA_CALL(hipMalloc((void **)&devStates, 64 * 64 *
                  sizeof(hiprandState)));
    }

    /* Setup prng states */
    if (useMRG) {
        setup_kernel<<<64, 64>>>(devMRGStates);
    }else if(usePHILOX)
    {
        setup_kernel<<<64, 64>>>(devPHILOXStates);
    }else {
        setup_kernel<<<64, 64>>>(devStates);
    }

    /* Generate and use pseudo-random  */
    for(i = 0; i < 50; i++) {
        if (useMRG) {
            generate_kernel<<<64, 64>>>(devMRGStates, sampleCount, devResults);
        }else if (usePHILOX){
            generate_kernel<<<64, 64>>>(devPHILOXStates, sampleCount, devResults);
	}else {
            generate_kernel<<<64, 64>>>(devStates, sampleCount, devResults);
        }
    }

    /* Copy device memory to host */
    CUDA_CALL(hipMemcpy(hostResults, devResults, 64 * 64 *
        sizeof(unsigned int), hipMemcpyDeviceToHost));

    /* Show result */
    total = 0;
    for(i = 0; i < 64 * 64; i++) {
        total += hostResults[i];
    }
    printf("Fraction with low bit set was %10.13f\n",
        (float)total / (64.0f * 64.0f * sampleCount * 50.0f));

    /* Set results to 0 */
    CUDA_CALL(hipMemset(devResults, 0, 64 * 64 *
              sizeof(unsigned int)));

    /* Generate and use uniform pseudo-random  */
    for(i = 0; i < 50; i++) {
        if (useMRG) {
            generate_uniform_kernel<<<64, 64>>>(devMRGStates, sampleCount, devResults);
        }else if(usePHILOX) {
            generate_uniform_kernel<<<64, 64>>>(devPHILOXStates, sampleCount, devResults);
	}else {
            generate_uniform_kernel<<<64, 64>>>(devStates, sampleCount, devResults);
        }
    }

    /* Copy device memory to host */
    CUDA_CALL(hipMemcpy(hostResults, devResults, 64 * 64 *
        sizeof(unsigned int), hipMemcpyDeviceToHost));

    /* Show result */
    total = 0;
    for(i = 0; i < 64 * 64; i++) {
        total += hostResults[i];
    }
    printf("Fraction of uniforms > 0.5 was %10.13f\n",
        (float)total / (64.0f * 64.0f * sampleCount * 50.0f));
    /* Set results to 0 */
    CUDA_CALL(hipMemset(devResults, 0, 64 * 64 *
              sizeof(unsigned int)));

    /* Generate and use normal pseudo-random  */
    for(i = 0; i < 50; i++) {
        if (useMRG) {
            generate_normal_kernel<<<64, 64>>>(devMRGStates, sampleCount, devResults);
        }else if(usePHILOX) {
            generate_normal_kernel<<<64, 64>>>(devPHILOXStates, sampleCount, devResults);
	}else {
            generate_normal_kernel<<<64, 64>>>(devStates, sampleCount, devResults);
        }
    }

    /* Copy device memory to host */
    CUDA_CALL(hipMemcpy(hostResults, devResults, 64 * 64 *
        sizeof(unsigned int), hipMemcpyDeviceToHost));

    /* Show result */
    total = 0;
    for(i = 0; i < 64 * 64; i++) {
        total += hostResults[i];
    }
    printf("Fraction of normals within 1 standard deviation was %10.13f\n",
        (float)total / (64.0f * 64.0f * sampleCount * 50.0f));

    /* Cleanup */
    if (useMRG) {
        CUDA_CALL(hipFree(devMRGStates));
    }else if(usePHILOX)
    {
        CUDA_CALL(hipFree(devPHILOXStates));
    }else {
        CUDA_CALL(hipFree(devStates));
    }
    CUDA_CALL(hipFree(devResults));
    free(hostResults);
    printf("^^^^ kernel_example PASSED\n");
    return EXIT_SUCCESS;
}
