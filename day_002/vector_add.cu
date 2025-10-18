#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <chrono>

#define N 10000000
#define BLOCK_SIZE 256 // 256 threads per block

// Returns the current time in seconds
double get_time(){
  using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

// Example:
// A = [1,2,3,4,5]
// B = [6,7,8,9,10]
// C = A + B =  [7,9,11,13,15]

// CPU vector_add
void vector_add_cpu(float *a, float *b, float *c, int n){
    for(int i=0; i<n; i++){
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for vector add_gpu
__global__ void vector_add_gpu(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x; //index of current head within block Eg: 

// Suppose: BLOCK_SIZE = 256

// You launched num_blocks = 10
// So you have 256 * 10 = 2560 threads total.

// Now, suppose you are:

// in block 2 (blockIdx.x = 2)

// thread 5 in that block (threadIdx.x = 5)

// Then:
// i = 2 * 256 + 5 = 517
// This thread will compute:

// c[517] = a[517] + b[517];

    if(i<n){
        c[i] = a[i] + b[i];
    }
}

// init vec with random vals
void init_vector(float *vec, int n){
    for(int i=0; i<n; i++){
        vec[i] = (float)rand() / RAND_MAX;
    }
}


int main(){
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c_cpu = (float *)malloc(size);
    h_c_gpu = (float *)malloc(size);

    // Initialize vectors
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy vectors from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    // how many blcok do you need to size BLOCK_SIZE
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // blocksize - no of threads inside the block
    // elements to add N = 1024, BLOCK_SIZE=256, num_blocks = 4  

    printf("perform warm up runs\n");
    for(int i=0; i<3; i++){
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize(); //Wait for compute device to finish    
    }

    // Benchmark CPU implementation
    printf("benchmarking CPU implementation\n");
    double cpu_total_time = 0.0;
    for(int i=0; i<20; i++){
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += (end_time - start_time);
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Print results
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Verify results (optional)
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        //floating point abs()
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
        printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, h_c_cpu[i], h_c_gpu[i]);
        correct = false;
        break;
    }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");
    free(h_a); free(h_b); free(h_c_cpu); free(h_c_gpu);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;

}

// OUTPUT
// perform warm up runs
// benchmarking CPU implementation
// Benchmarking GPU implementation...
// CPU average time: 30.530973 milliseconds
// GPU average time: 0.230527 milliseconds
// Speedup: 132.439842x
// Results are correct