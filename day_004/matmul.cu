#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256 // Number of rows in A and C
#define K 512 // Number of columns in A and rows in B
#define N 256 // Number of columns in B and C
#define BLOCK_SIZE 32
// Example 3x2 @ 2x4 = 3x4
//         MxK  KxN    MxN
// A = [[1 2],
//      [3 4],
//      [5 6]]

// B = [[7  8  9 10],
//.     [11 12 13 14]]

// row major formula = A[row * n_cols + col]

// C = A * B = [[1*7 + 2*11, 1*8 + 2*12, 1*9 + 2*13, 1*10 + 2*14],
//              [3*7 + 4*11, 3*8 + 4*12, 3*9 + 4*13, 3*10 + 4*14],
//              [5*7 + 6*11, 5*8 + 6*12, 5*9 + 6*13, 5*10 + 6*14]]

// C = [[29, 32, 35, 38],
//      [65, 72, 79, 86],
//      [101, 112, 123, 134]]


__global__ void matmul(float*A, float*B, float*C, int m, int k, int n){

    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= m || col >= n) return;
    float acc = 0.0f;
    for (int i = 0; i<k; i++){
        acc += A[row * k + i] * B[ i * n + col]; //row indexed by col - N
    }
    C[row * n + col] = acc;
}

void init_matrix(float *mat, int rows, int cols){

    for(int i=0; i<rows*cols; i++){
        mat[i] = (float)rand() / RAND_MAX;
    }
}

void print_matrix(float *mat, int rows, int cols){
    for(int i=0l i<rows; i++){
        for(int j=0; j<cols; j++){
            printf("%f ", mat[i*cols + j]);
        }
        printf("\n");
    }
}

int main(){
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size_A = M*K*sizeof(float);
    int size_B = K*N*sizeof(float);
    int size_C = M*N*sizeof(float);

    h_a = (float*)malloc(size_A);
    h_b = (float*)malloc(size_B);
    h_c = (float*)malloc(size_C);

    init_matrix(h_a, M, K);
    init_matrix(h_b, K, N);

    cudaMalloc(&d_a, size_A);
    cudaMalloc(&d_b, size_B);
    cudaMalloc(&d_c, size_C);

    cudaMemcpy(d_a, h_a, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
                    (N+BLOCK_SIZE-1)/BLOCK_SIZE, 
                    (M+BLOCK_SIZE-1)/BLOCK_SIZE
                );

    matmul<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size_C, cudaMemcpyDeviceToHost);

    print_matrix(h_c, M, N);

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}


