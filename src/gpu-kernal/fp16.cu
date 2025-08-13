#include<stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fp16.cu.h"

__global__ void fp16ToFp32Gpu(uint16_t* fp16_buffer, float* fp32_buffer, int val_num) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (thread_id < val_num) {
        __half fp16 = *(__half*)((fp16_buffer + thread_id));
        *(fp32_buffer + thread_id) = __half2float(fp16);
        thread_id += stride;
    }
}

void lanuchFp16ToFp32Gpu(uint16_t* fp16_buffer, float* fp32_buffer, int val_num) {
    int block_size = 512;
    int total_num = val_num;
    if (total_num > 10000) {
        total_num = 10000;
    }
    int grid_size  = (total_num + block_size - 1) / block_size;
    fp16ToFp32Gpu<<<grid_size, block_size>>>(fp16_buffer, fp32_buffer, val_num);
    cudaDeviceSynchronize();
}

int lanuchFp16ToFp32GpuInplace(uint16_t* fp16_buffer, int val_num) {
    int cur_end = val_num - 1;
    int block_size = 512;
    int global_index = 0;
    while( cur_end >= 0) {
        global_index += 1;
        int cur_start = cur_end / 2 + cur_end % 2;
        int cur_num = (cur_end - cur_start + 1);
        int total_num = 10000;
        if (total_num > cur_num) {
            total_num = cur_num;
        }
        int grid_size = (total_num + block_size - 1) / block_size;
        uint16_t* fp16_buf = fp16_buffer + cur_start;
        float* fp32_buf = (float*)(fp16_buffer + cur_start * 2);
        fp16ToFp32Gpu<<<grid_size, block_size>>>(fp16_buf, fp32_buf, cur_num);
        cudaDeviceSynchronize();
        cur_end = cur_start - 1;
        //printf("Inplace_fp16_fp32, global_index: %d, cur_num: %d\n", global_index, cur_num);
        if (cur_end < 1000) {
            //TODO move to cpu 
            return cur_end;
        }
    }
    return -1;
}
