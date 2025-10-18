# include <stdio.h>

// Threads
// The smallest work‐unit. Each thread has its own registers and private local memory.

// Thread-Blocks (blockDim, blockIdx)
// A group of up to 1 024 threads (organized in a 1D, 2D or 3D array). Threads within a block can:

// share data via fast shared memory

// synchronize with __syncthreads()

// Grid (gridDim)
// An array of thread-blocks, all running the same kernel. Blocks cannot synchronize or share memory with each other.
// Inside a __global__ (device) function you have four read-only structs:

// blockDim : dimensions of your block (how many threads in each block)

// gridDim : dimensions of your grid (how many blocks in each grid)

// threadIdx : (x,y,z) coordinate of this thread within its block

// blockIdx : (x,y,z) coordinate of this block within the grid

__global__ void whoami(void){
    int block_id = 
        blockIdx.x +    // apartment number on this floor (points across)
        blockIdx.y * gridDim.x +    // floor number in this building (rows high)
        blockIdx.z * gridDim.x * gridDim.y;   // building number in this city (panes deep)
        printf("")

    int block_offset = 
        block_id * //no of blocks before this
        blockDim.x * blockDim.y * blockDim.z; //total threads per block
    
    int thread_offset = 
        threadIdx.x + 
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    int id = block_offset + thread_offset;

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);

    }

int main(int argc, char **argv){
    const int b_x = 2, b_y = 3, b_z = 4;
    const int t_x = 4, t_y = 4, t_z = 4; // the max warp size is 32, so 
    // we will get 2 warp of 32 threads per block

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;

    printf("%d blocks/grid\n", blocks_per_grid);
    printf("%d threads/block\n", threads_per_block);
    printf("%d total threads\n", blocks_per_grid * threads_per_block);

    dim3 blocksPerGrid(b_x, b_y, b_z); //3d cude of shape 2*3*4 = 24
    dim3 threadsPerBlock(t_x, t_y, t_z); //3d cude of shape 4*4*4 = 64

    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize(); // wait for all threads to finish
}