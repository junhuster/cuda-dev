#include<stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fp16.cu.h"

const int BLOCK_SIZE = 512;

__global__ void warmup(uint64_t* input_buffer, int val_num, uint64_t* block_output) {  
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t* my_data = input_buffer + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    if (index >= val_num) return;
    uint64_t warmup_num = 0;
    for (int s = 1; s < blockDim.x; s = s * 2) {
        if (tid % (2*s) == 0) {
            warmup_num = my_data[tid] + my_data[tid + s];
        }
        __syncthreads();
    }
}

__global__ void reduceSumBase(uint64_t* input_buffer, int val_num, uint64_t* block_output) {  
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t* my_data = input_buffer + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    if (index >= val_num) return;
    for (int s = 1; s < blockDim.x; s = s * 2) {
        if (tid % (2*s) == 0) {
            my_data[tid] += my_data[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_output[blockIdx.x] = my_data[0];
    }
}

__global__ void reduceSumSmallDataSet(uint64_t* input_buffer, int val_num, uint64_t* output) {
    uint64_t num = 0;
    if (threadIdx.x == 0) {
        for (int i = 0; i < val_num; i++) {
            num += *(input_buffer + i);
        }
        *output = num;
    }
}

__global__ void reduceSumShamem(uint64_t* input_buffer, int val_num, uint64_t* block_output) {  
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= val_num) return;
    __shared__ uint64_t my_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    my_data[tid] = *(input_buffer + index);
    __syncthreads();
    for (int s = 1; s < blockDim.x; s = s * 2) {
        if (tid % (2*s) == 0) {
            my_data[tid] += my_data[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_output[blockIdx.x] = my_data[0];
    }
}

__global__ void reduceSumWarpOpt(uint64_t* input_buffer, int val_num, uint64_t* block_output) {  
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= val_num) return;
    __shared__ uint64_t my_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    my_data[tid] = *(input_buffer + index);
    __syncthreads();
    for (int s = 1; s < blockDim.x; s = s * 2) {
        int idx = tid * 2 * s;
        if (idx < blockDim.x) {
            my_data[idx] += my_data[idx + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_output[blockIdx.x] = my_data[0];
    }
}
  
__global__ void reduceSumBankOpt(uint64_t* input_buffer, int val_num, uint64_t* block_output) {  
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= val_num) return;
    __shared__ uint64_t my_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    my_data[tid] = *(input_buffer + index);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s = s / 2) {
        if (tid < s) {
            my_data[tid] = my_data[tid] + my_data[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_output[blockIdx.x] = my_data[0];
    }
}

void lanuchWarmup(uint64_t* input_buffer, int val_num, int block_size, uint64_t* block_output) {
    int grid_size = (val_num + block_size - 1) / block_size;
    warmup<<<grid_size, block_size>>>(input_buffer, val_num, block_output);
    cudaDeviceSynchronize();
}

void lanuchReduceSumBase(uint64_t* input_buffer, int val_num, int block_size, uint64_t* block_output) {
    int grid_size = (val_num + block_size - 1) / block_size;
    reduceSumBase<<<grid_size, block_size>>>(input_buffer, val_num, block_output);
    cudaDeviceSynchronize();
}

void lanuchReduceSumBase2(uint64_t* input_buffer, int val_num, int block_size, uint64_t* block_output, const ::cudaStream_t stream) {
    int grid_size = (val_num + block_size - 1) / block_size;
    reduceSumBase<<<grid_size, block_size, 0, stream>>>(input_buffer, val_num, block_output);
    int old_grid_dim = grid_size;
    grid_size = (old_grid_dim + block_size - 1) / block_size;
    reduceSumBase<<<grid_size, block_size, 0, stream>>>(block_output, old_grid_dim, input_buffer);
    reduceSumSmallDataSet<<<1, 32, 0, stream>>>(input_buffer, grid_size, block_output);
}

void lanuchReduceSumShamem(uint64_t* input_buffer, int val_num, int block_size, uint64_t* block_output, const ::cudaStream_t stream = 0) {
    uint64_t shamem_size = BLOCK_SIZE * sizeof(uint64_t);
    int grid_size = (val_num + block_size - 1) / block_size;
    reduceSumShamem<<<grid_size, block_size, shamem_size, stream>>>(input_buffer, val_num, block_output);
    int old_grid_dim = grid_size;
    grid_size = (old_grid_dim + block_size - 1) / block_size;
    reduceSumShamem<<<grid_size, block_size, shamem_size, stream>>>(block_output, old_grid_dim, input_buffer);
    reduceSumSmallDataSet<<<1, 32, 0, stream>>>(input_buffer, grid_size, block_output);
}

void lanuchReduceSumWarpOpt(uint64_t* input_buffer, int val_num, int block_size, uint64_t* block_output, const ::cudaStream_t stream = 0) {
    uint64_t shamem_size = BLOCK_SIZE * sizeof(uint64_t);
    int grid_size = (val_num + block_size - 1) / block_size;
    reduceSumWarpOpt<<<grid_size, block_size, shamem_size, stream>>>(input_buffer, val_num, block_output);
    int old_grid_dim = grid_size;
    grid_size = (old_grid_dim + block_size - 1) / block_size;
    reduceSumWarpOpt<<<grid_size, block_size, shamem_size, stream>>>(block_output, old_grid_dim, input_buffer);
    reduceSumSmallDataSet<<<1, 32, 0, stream>>>(input_buffer, grid_size, block_output);
}

void lanuchReduceSumBankOpt(uint64_t* input_buffer, int val_num, int block_size, uint64_t* block_output, const ::cudaStream_t stream = 0) {
    uint64_t shamem_size = BLOCK_SIZE * sizeof(uint64_t);
    int grid_size = (val_num + block_size - 1) / block_size;
    reduceSumBankOpt<<<grid_size, block_size, shamem_size, stream>>>(input_buffer, val_num, block_output);
    int old_grid_dim = grid_size;
    grid_size = (old_grid_dim + block_size - 1) / block_size;
    reduceSumBankOpt<<<grid_size, block_size, shamem_size, stream>>>(block_output, old_grid_dim, input_buffer);
    reduceSumSmallDataSet<<<1, 32, 0, stream>>>(input_buffer, grid_size, block_output);
}