#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <chrono>
#include <cstdio>
__device__ int completedThreads = 0;
__global__ void fill_kernel(uchar4* ptr, int w, int h,int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    unsigned int hash = (x * 73856093) ^ (y * 19349663) ^ seed;
    // 第一个数 R
    hash = hash * 1103515245 + 12345;
    unsigned char r = (hash % 255) + 1;
    
    // 第二个数 G（继续使用哈希链）
    hash = hash * 1103515245 + 12345;
    unsigned char g = (hash % 255) + 1;
    
    // 第三个数 B
    hash = hash * 1103515245 + 12345;
    unsigned char b = (hash % 255) + 1;

    // ptr[idx] = make_uchar4(169, 100, 50, 255);
    ptr[idx] = make_uchar4(r, g, b, 255);
        // printf("--------------------\r\n");
}
#include <iostream>
void launch_cuda_kernel(uchar4* devPtr, int width, int height, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,(height + block.y - 1) / block.y);
    // 使用高精度时间作为种子
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 generator(static_cast<unsigned int>(seed));
    std::uniform_int_distribution<int> distribution(1, 1000);
    int random_num = distribution(generator);
    fill_kernel<<<grid, block,0,stream>>>(devPtr, width, height,random_num);
    // cudaError_t err = cudaGetLastError();
    // printf("launch_cuda_kernel:  %s\r\n",cudaGetErrorString(err));
    // cudaStreamSynchronize(stream);
}
