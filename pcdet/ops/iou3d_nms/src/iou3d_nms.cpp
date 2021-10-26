/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "iou3d_nms.h"

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8; // NMS中每个block中的线程 8x8=64


void boxesoverlapLauncher(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_overlap);
void boxesioubevLauncher(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_iou);
void nmsLauncher(const float *boxes, unsigned long long * mask, int boxes_num, float nms_overlap_thresh); // 对应核函数声明
void nmsNormalLauncher(const float *boxes, unsigned long long * mask, int boxes_num, float nms_overlap_thresh);


int boxes_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_overlap: (N, M)

    CHECK_INPUT(boxes_a);
    CHECK_INPUT(boxes_b);
    CHECK_INPUT(ans_overlap);

    int num_a = boxes_a.size(0);
    int num_b = boxes_b.size(0);

    const float * boxes_a_data = boxes_a.data<float>();
    const float * boxes_b_data = boxes_b.data<float>();
    float * ans_overlap_data = ans_overlap.data<float>();

    boxesoverlapLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_overlap_data);

    return 1;
}


int boxes_iou_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_overlap: (N, M)
    CHECK_INPUT(boxes_a);
    CHECK_INPUT(boxes_b);
    CHECK_INPUT(ans_iou);

    int num_a = boxes_a.size(0);
    int num_b = boxes_b.size(0);

    const float * boxes_a_data = boxes_a.data<float>();
    const float * boxes_b_data = boxes_b.data<float>();
    float * ans_iou_data = ans_iou.data<float>();

    boxesioubevLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_iou_data);

    return 1;
}


int nms_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params keep: (N)
    // 首先对输入向量进行检查
    CHECK_INPUT(boxes);
    CHECK_CONTIGUOUS(keep);

    // 获取tensor中的数据指针，此时数据本身就在device上，无需从host拷贝到device上
    int boxes_num = boxes.size(0);
    const float * boxes_data = boxes.data<float>();
    long * keep_data = keep.data<long>();

    // 计算块的数量
    // 由于SM的基本执行单元是包含32个线程的线程束，所以block大小一般要设置为32的倍数
    const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS); // 向上取整 4096/64=64

    unsigned long long *mask_data = NULL;
    // 在device上为mask_data申请一定大小的显存
    // cudaError_t cudaMalloc(void** devPtr, size_t size);
    CHECK_ERROR(cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long)));
    // 调用CUDA函数
    nmsLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

    // unsigned long long mask_cpu[boxes_num * col_blocks];
    // unsigned long long *mask_cpu = new unsigned long long [boxes_num * col_blocks];

    std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks); // 4096 x 64

    // printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
    // cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
    CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data, boxes_num * col_blocks * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));

    // 释放device内存
    cudaFree(mask_data);

    unsigned long long remv_cpu[col_blocks];
    // 长为64的unsigned long long数组，初始值为0，表示全部保留
    // 这4096个标记位用于记录哪些候选框已经被剔除
    memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long)); 
    // 保留下来的个数
    int num_to_keep = 0;

    for (int i = 0; i < boxes_num; i++){
        // 把4096个候选框分成64组，每64个一组，第nblock组的第inblock个候选框就是第i个候选框
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS; 
        // 如果第i个框的remv_cpu标记为1,则直接跳过该框，否则继续
        if (!(remv_cpu[nblock] & (1ULL << inblock))){
            // 记录被保留box的id
            keep_data[num_to_keep++] = i;
            // mask_cpu的总长为4096*64,分为4096段，第i段有64个unsigned long long整数，
            // 记录第i个候选框与其他所有候选框之间的信息，p指向的是第i段的起始位置
            unsigned long long *p = &mask_cpu[0] + i * col_blocks;
            // 遍历后面的64个整数，总共4096个标记位
            for (int j = nblock; j < col_blocks; j++){
                // 因为第i个候选框已经保留下来，所以与第i个候选框重复的候选框都标记为1,即去除
                remv_cpu[j] |= p[j];
            }
        }
    }
    if ( cudaSuccess != cudaGetLastError() ) printf( "Error!\n" );
    // 前num_to_keep个数就是保留下来的候选框的下标
    return num_to_keep;
}


int nms_normal_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params keep: (N)

    CHECK_INPUT(boxes);
    CHECK_CONTIGUOUS(keep);

    int boxes_num = boxes.size(0);
    const float * boxes_data = boxes.data<float>();
    long * keep_data = keep.data<long>();

    const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

    unsigned long long *mask_data = NULL;
    CHECK_ERROR(cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long)));
    nmsNormalLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

    // unsigned long long mask_cpu[boxes_num * col_blocks];
    // unsigned long long *mask_cpu = new unsigned long long [boxes_num * col_blocks];
    std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

//    printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
    CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data, boxes_num * col_blocks * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));

    cudaFree(mask_data);

    unsigned long long remv_cpu[col_blocks];
    memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long));

    int num_to_keep = 0;

    for (int i = 0; i < boxes_num; i++){
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS;

        if (!(remv_cpu[nblock] & (1ULL << inblock))){
            keep_data[num_to_keep++] = i;
            unsigned long long *p = &mask_cpu[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++){
                remv_cpu[j] |= p[j];
            }
        }
    }
    if ( cudaSuccess != cudaGetLastError() ) printf( "Error!\n" );

    return num_to_keep;
}


