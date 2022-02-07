/*
Stacked-batch-data version of point grouping, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "group_points_gpu.h"


__global__ void group_points_grad_kernel_stack(int B, int M, int C, int N, int nsample,
    const float *grad_out, const int *idx, const int *idx_batch_cnt, const int *features_batch_cnt, float *grad_features) {
    // :param grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
    // :return:
    //     grad_features: (N1 + N2 ..., C) gradient of the features
    int index = blockIdx.x * blockDim.x + threadIdx.x; // 计算线程索引
    int sample_idx = index % nsample; // 计算采样点索引
    int C_idx = (index / nsample) % C; // 计算特征索引
    int pt_idx = (index / nsample / C); // 计算网格点索引

    if (pt_idx >= M || C_idx >= C || sample_idx >= nsample) return;
    
    // 计算当前网格点所属batch index
    int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
    for (int k = 1; k < B; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += idx_batch_cnt[k];
        bs_idx = k;
    }
    // 根据网格点所属batch，计算features起始batch的地址
    int features_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) features_batch_start_idx += features_batch_cnt[k];

    grad_out += pt_idx * C * nsample + C_idx * nsample + sample_idx; // 计算梯度输出地址
    idx += pt_idx * nsample + sample_idx;
    grad_features += (features_batch_start_idx + idx[0]) * C + C_idx; // 计算特征地址
    
    // 因为特征聚合使得特征顺序变换，因此梯度的反向传播也要寻找对应位置的梯度
    atomicAdd(grad_features, grad_out[0]); // 将该特征的梯度放到对应特征上
}

void group_points_grad_kernel_launcher_stack(int B, int M, int C, int N, int nsample,
    const float *grad_out, const int *idx, const int *idx_batch_cnt, const int *features_batch_cnt, float *grad_features) {
    // :param grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
    // :return:
    //     grad_features: (N1 + N2 ..., C) gradient of the features

    cudaError_t err;
    // dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 blocks(DIVUP(M * C * nsample, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    group_points_grad_kernel_stack<<<blocks, threads>>>(B, M, C, N, nsample, grad_out, idx, idx_batch_cnt, features_batch_cnt, grad_features);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void group_points_kernel_stack(int B, int M, int C, int nsample,
    const float *features, const int *features_batch_cnt, const int *idx, const int *idx_batch_cnt, float *out) {
    // :param features: (N1 + N2 ..., C) tensor of features to group
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
    // :return:
    //     output: (M1 + M2, C, nsample) tensor
    int index = blockIdx.x * blockDim.x + threadIdx.x; // 计算线程索引
    // 这里计算的实际上是输出索引的3个坐标（由同一个线程的index计算）
    int sample_idx = index % nsample; // 计算采样点索引
    int C_idx = (index / nsample) % C; // 计算特征索引
    int pt_idx = (index / nsample / C); // 计算网格点索引

    if (pt_idx >= M || C_idx >= C || sample_idx >= nsample) return;

    int bs_idx = 0, pt_cnt = idx_batch_cnt[0]; // 获取batch size索引，获取第一帧特征点数量
    // 计算当前网格点所属batch index
    for (int k = 1; k < B; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += idx_batch_cnt[k];
        bs_idx = k;
    }

    int features_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) features_batch_start_idx += features_batch_cnt[k]; // 根据网格点所属batch，计算features起始batch的地址
    features += features_batch_start_idx * C; // 计算该帧特征起始地址

    idx += pt_idx * nsample + sample_idx; // 计算特征索引地址（按照batch减去对应起始地址，这里已经是帧内相对索引）
    int in_idx = idx[0] * C + C_idx; // 计算feature在帧内的地址
    int out_idx = pt_idx * C * nsample + C_idx * nsample + sample_idx; // 计算输出的地址

    out[out_idx] = features[in_idx]; // 将features赋值到输出位置上
}


void group_points_kernel_launcher_stack(int B, int M, int C, int nsample,
    const float *features, const int *features_batch_cnt, const int *idx, const int *idx_batch_cnt, float *out) {
    // :param features: (N1 + N2 ..., C) tensor of features to group
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
    // :return:
    //     output: (M1 + M2, C, nsample) tensor

    cudaError_t err;
    dim3 blocks(DIVUP(M * C * nsample, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row) 221184 * 32 * 16 每个线程处理一个特征
    dim3 threads(THREADS_PER_BLOCK);

    group_points_kernel_stack<<<blocks, threads>>>(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, out);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
