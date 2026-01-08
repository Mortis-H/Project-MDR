
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void cube(float * d_out, float * d_in) {
	int index = threadIdx.x;
	float f = d_in[index];
	d_out[index] = f * f * f;
}

__global__ void square(float * d_out, float * d_in) {
	int index = threadIdx.x;
	float f = d_in[index];
	d_out[index] = f * f;
}

int main(int argc, char ** argv) {
	const int ARRAY_SIZE = 25;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float h1_in[ARRAY_SIZE], h2_in[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		h1_in[i] = h2_in[i] = float(i);
	}
	float h1_out[ARRAY_SIZE], h2_out[ARRAY_SIZE];

	// declare GPU memory pointers
	float * d1_in, *d2_in;
	float * d1_out, *d2_out;

	hipSetDevice(0);
	// allocate GPU memory
	hipMalloc((void**) &d1_in, ARRAY_BYTES);
	hipMalloc((void**) &d1_out, ARRAY_BYTES);

	hipSetDevice(1);
	hipMalloc((void**) &d2_in, ARRAY_BYTES);
	hipMalloc((void**) &d2_out, ARRAY_BYTES);

	hipSetDevice(0);
	// transfer the array to the GPU
	hipMemcpy(d1_in, h1_in, ARRAY_BYTES, hipMemcpyHostToDevice);

	hipSetDevice(1);
	hipMemcpy(d2_in, h2_in, ARRAY_BYTES, hipMemcpyHostToDevice);

	hipSetDevice(0);
	// launch the kernel
	cube<<<1, ARRAY_SIZE>>>(d1_out, d1_in);

	hipSetDevice(1);
	square<<<1, ARRAY_SIZE>>>(d2_out, d2_in);

	hipSetDevice(0);
	// copy back the result array to the CPU
	hipMemcpy(h1_out, d1_out, ARRAY_BYTES, hipMemcpyDeviceToHost);

	hipSetDevice(1);
	hipMemcpy(h2_out, d2_out, ARRAY_BYTES, hipMemcpyDeviceToHost);

	// print out the resulting array
	for (int i =0; i < ARRAY_SIZE; i++) {
		printf("%f", h1_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}
	printf("\n---------------------------------------------------\n\n");
	// print out the resulting array
	for (int i =0; i < ARRAY_SIZE; i++) {
		printf("%f", h2_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}

	hipSetDevice(0);
	hipFree(d1_in);
	hipFree(d1_out);

	hipSetDevice(1);
	hipFree(d2_in);
	hipFree(d2_out);

	return 0;
}
