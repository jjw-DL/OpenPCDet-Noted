/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <stdio.h>
#define THREADS_PER_BLOCK 16
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// #define DEBUG
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8; // 64
const float EPS = 1e-8;

/// @brief 定义Point结构体
struct Point {
    // 成员变量
    float x, y;
    // 构造函数
    __device__ Point() {}
    __device__ Point(double _x, double _y){
        x = _x, y = _y;
    }
    // set函数
    __device__ void set(float _x, float _y){
        x = _x; y = _y;
    }
    // 重载加法操作
    __device__ Point operator +(const Point &b)const{
        return Point(x + b.x, y + b.y);
    }
    // 重载减法操作
    __device__ Point operator -(const Point &b)const{
        return Point(x - b.x, y - b.y);
    }
};

/// @brief 点的叉乘
__device__ inline float cross(const Point &a, const Point &b){
    return a.x * b.y - a.y * b.x;
}
/// @brief 向量叉乘
__device__ inline float cross(const Point &p1, const Point &p2, const Point &p0){
    // a x b = (x1 * y2) - (x2 * y1)
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

/// @brief 检查box的交叉: p1和p2的连线和q1和q2的连线相交才能返回True
__device__ int check_rect_cross(const Point &p1, const Point &p2, const Point &q1, const Point &q2){
    int ret = min(p1.x,p2.x) <= max(q1.x,q2.x) &&
              min(q1.x,q2.x) <= max(p1.x,p2.x) &&
              min(p1.y,p2.y) <= max(q1.y,q2.y) &&
              min(q1.y,q2.y) <= max(p1.y,p2.y);
    return ret;
}

/// @brief 检查点是否在box内
__device__ inline int check_in_box2d(const float *box, const Point &p){
    //params: (7) [x, y, z, dx, dy, dz, heading]
    const float MARGIN = 1e-2;
    // 将点从lidar坐标系变换到局部坐标系下 p' = R * (p - t)
    float center_x = box[0], center_y = box[1];
    float angle_cos = cos(-box[6]), angle_sin = sin(-box[6]);  // rotate the point in the opposite direction of box
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;
    // 在正交坐标系下只需要判断点和一半长宽的关系即可算出点是否在box内
    return (fabs(rot_x) < box[3] / 2 + MARGIN && fabs(rot_y) < box[4] / 2 + MARGIN);
}

/// @brief 计算两条直线交点坐标
__device__ inline int intersection(const Point &p1, const Point &p0, const Point &q1, const Point &q0, Point &ans){
    // 参考博客:
    // https://www.cnblogs.com/xpvincent/p/5208994.html
    // https://blog.csdn.net/u011089570/article/details/79040948
    // 首先判断两条直线是否相交
    // fast exclusion
    if (check_rect_cross(p0, p1, q0, q1) == 0) return 0;

    // check cross standing
    // 求4个点够构成的4个三角形的面积，由于叉乘关系，会有正负之分
    // 位于同侧的点符号相同，异侧符号相反
    float s1 = cross(q0, p1, p0); 
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

    // calculate intersection of two lines
    // 求两线段交点
    float s5 = cross(q1, p1, p0);
    if(fabs(s5 - s1) > EPS){
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

    }
    else{
        // 表示直线方程的系数ax + by + c = 0,克莱姆法则求解方程的解
        // Dj是把D中第j列元素对应地换成常数项而其余各列保持不变所得到的行列式
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return 1;
}

/// @brief 绕中心点旋转
__device__ inline void rotate_around_center(const Point &center, const float angle_cos, const float angle_sin, Point &p){
    // 先转换局部坐标，再加上中心点坐标
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y);
}

/// @brief 比较两个点的旋转角
__device__ inline int point_cmp(const Point &a, const Point &b, const Point &center){
    return atan2(a.y - center.y, a.x - center.x) > atan2(b.y - center.y, b.x - center.x);
}

/// @brief 计算交集多边形的面积
/// 1.求两个box的四个角点坐标，先平移，然后旋转
/// 2.求所有线的交点
/// 3.求交集多边形中属于box的角点
/// 4.根据对交集多边形中心点对其角点进行排序，使其满足逆时针方向
/// 5.根据叉乘几何意义，求取交集多边形面积
__device__ inline float box_overlap(const float *box_a, const float *box_b){
    // params box_a: [x, y, z, dx, dy, dz, heading]
    // params box_b: [x, y, z, dx, dy, dz, heading]

    // 1.求两个box的四个角点坐标，先平移，然后旋转
    float a_angle = box_a[6], b_angle = box_b[6];
    float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2, a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
    float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half; // 左和下边界
    float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half; // 右和上边界
    float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half; 
    float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;

    Point center_a(box_a[0], box_a[1]); // 中心点坐标
    Point center_b(box_b[0], box_b[1]);

#ifdef DEBUG
    printf("a: (%.3f, %.3f, %.3f, %.3f, %.3f), b: (%.3f, %.3f, %.3f, %.3f, %.3f)\n", a_x1, a_y1, a_x2, a_y2, a_angle,
           b_x1, b_y1, b_x2, b_y2, b_angle);
    printf("center a: (%.3f, %.3f), b: (%.3f, %.3f)\n", center_a.x, center_a.y, center_b.x, center_b.y);
#endif

    Point box_a_corners[5];
    box_a_corners[0].set(a_x1, a_y1); // 左下
    box_a_corners[1].set(a_x2, a_y1); // 右下
    box_a_corners[2].set(a_x2, a_y2); // 右上
    box_a_corners[3].set(a_x1, a_y2); // 左上

    Point box_b_corners[5];
    box_b_corners[0].set(b_x1, b_y1);
    box_b_corners[1].set(b_x2, b_y1);
    box_b_corners[2].set(b_x2, b_y2);
    box_b_corners[3].set(b_x1, b_y2);

    // get oriented corners
    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++){
#ifdef DEBUG
        printf("before corner %d: a(%.3f, %.3f), b(%.3f, %.3f) \n", k, box_a_corners[k].x, box_a_corners[k].y, box_b_corners[k].x, box_b_corners[k].y);
#endif  
        // 将四个角点绕中心点旋转
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
#ifdef DEBUG
        printf("corner %d: a(%.3f, %.3f), b(%.3f, %.3f) \n", k, box_a_corners[k].x, box_a_corners[k].y, box_b_corners[k].x, box_b_corners[k].y);
#endif
    }
    
    // 2.求所有线的交点
    // 将box_a_corners的最后一个点赋值为第一个点，五个点才能组成4条线
    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // get intersection of lines 两个box各4条线，一共16种情况
    Point cross_points[16];
    Point poly_center; // 记录多边形中心点
    int cnt = 0, flag = 0; // 记录中心点个数

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            flag = intersection(box_a_corners[i + 1], box_a_corners[i], box_b_corners[j + 1], box_b_corners[j], cross_points[cnt]);
            if (flag){
                poly_center = poly_center + cross_points[cnt];
                cnt++;
#ifdef DEBUG
                // 打印交点和交线
                printf("Cross points (%.3f, %.3f): a(%.3f, %.3f)->(%.3f, %.3f), b(%.3f, %.3f)->(%.3f, %.3f) \n",
                    cross_points[cnt - 1].x, cross_points[cnt - 1].y,
                    box_a_corners[i].x, box_a_corners[i].y, box_a_corners[i + 1].x, box_a_corners[i + 1].y,
                    box_b_corners[i].x, box_b_corners[i].y, box_b_corners[i + 1].x, box_b_corners[i + 1].y);
#endif
            }
        }
    }

    // 3.求交集多边形中属于box的角点
    // check corners
    // 分别检查四个角点在对方box内
    for (int k = 0; k < 4; k++){
        if (check_in_box2d(box_a, box_b_corners[k])){
            // 如果角点在box内则将其加入中心点
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;
#ifdef DEBUG
                printf("b corners in a: corner_b(%.3f, %.3f)", cross_points[cnt - 1].x, cross_points[cnt - 1].y);
#endif
        }
        if (check_in_box2d(box_b, box_a_corners[k])){
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
#ifdef DEBUG
                printf("a corners in b: corner_a(%.3f, %.3f)", cross_points[cnt - 1].x, cross_points[cnt - 1].y);
#endif
        }
    }

    // 4.根据对交集多边形中心点对其角点进行排序，使其满足逆时针方向
    // 计算多边形的平均中心点
    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // sort the points of polygon（冒泡排序）
    Point temp;
    for (int j = 0; j < cnt - 1; j++){
        for (int i = 0; i < cnt - j - 1; i++){
            // 比较的是两个点和中心点连线与坐标轴的夹角大小(从小到大排序)
            if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)){
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

#ifdef DEBUG
    // 打印交点个数和交点坐标
    printf("cnt=%d\n", cnt);
    for (int i = 0; i < cnt; i++){
        printf("All cross point %d: (%.3f, %.3f)\n", i, cross_points[i].x, cross_points[i].y);
    }
#endif

    // 5.根据叉乘几何意义，求取交集多边形面积
    // 由于夹角逐渐增大，与第一个点构成的三角形面积一定是在同侧的符号相同
    // get the overlap areas
    float area = 0;
    for (int k = 0; k < cnt - 1; k++){
        area += cross(cross_points[k] - cross_points[0], cross_points[k + 1] - cross_points[0]);
    }

    return fabs(area) / 2.0;
}

/// @brief 获取bev视角iou
__device__ inline float iou_bev(const float *box_a, const float *box_b){
    // params box_a: [x, y, z, dx, dy, dz, heading]
    // params box_b: [x, y, z, dx, dy, dz, heading]
    // 计算两个box的面积
    float sa = box_a[3] * box_a[4];
    float sb = box_b[3] * box_b[4];
    // 计算交集面积
    float s_overlap = box_overlap(box_a, box_b);
    // iou计算
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

/// @brief 计算boxes_overlap的核函数
/// @param num_a: boxes_a的数量
/// @param boxes_a: boxes_a的指针
/// @param num_b: boxes_b的数量
/// @param boxes_b: boxes_b的指针
/// @param ans_overlap: overlap结果指针
__global__ void boxes_overlap_kernel(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_overlap){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    // 1.分配线程index，二维线程和二维块，类似两个for循环
    // 核函数和C函数明显的不同是循环体消失，内置的线程坐标变量替换了数组索引
    const int a_idx = blockIdx.y * THREADS_PER_BLOCK + threadIdx.y; // row
    const int b_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x; // col
    // 2.这里类似for循环中的中间条件判断
    if (a_idx >= num_a || b_idx >= num_b){
        return;
    }
    // 3.for循环的逻辑
    const float * cur_box_a = boxes_a + a_idx * 7; // 起点+偏移量，根据线程索引计算内存首地址
    const float * cur_box_b = boxes_b + b_idx * 7;
    float s_overlap = box_overlap(cur_box_a, cur_box_b); // 计算两个box的交集面积
    // 4.根据row和col的索引找到全局线性内存索引并写入
    ans_overlap[a_idx * num_b + b_idx] = s_overlap; // 线性全局内存偏移量 = y * row + col
}

/// @brief bev视角求box_iou的核函数
/// @param num_a: boxes_a的数量
/// @param boxes_a: boxes_a的指针
/// @param num_b: boxes_b的数量
/// @param boxes_b: boxes_b的指针
/// @param ans_iou: iou结果指针
__global__ void boxes_iou_bev_kernel(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_iou){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    const int a_idx = blockIdx.y * THREADS_PER_BLOCK + threadIdx.y;
    const int b_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    if (a_idx >= num_a || b_idx >= num_b){
        return;
    }

    const float * cur_box_a = boxes_a + a_idx * 7;
    const float * cur_box_b = boxes_b + b_idx * 7;
    float cur_iou_bev = iou_bev(cur_box_a, cur_box_b);
    ans_iou[a_idx * num_b + b_idx] = cur_iou_bev;
}


/// @brief nms核函数
/// @param boxes_num: box的数量
/// @param nms_overlap_thresh: nms阈值
/// @param boxes: box指针
/// @param mask: 记录box之间的iou关系，4096 × 64（4096/64）的矩阵 
/// 每连续的64个0、1刚好构成一个无符号的64位整数（unsigned long long，
/// 只用一个整数表示即可，这样内存减少64倍
__global__ void nms_kernel(const int boxes_num, const float nms_overlap_thresh,
                           const float *boxes, unsigned long long *mask){
    //params: boxes (N, 7) [x, y, z, dx, dy, dz, heading]
    //params: mask (N, N/THREADS_PER_BLOCK_NMS)
    // 矩阵的行号和列号(block级)
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    // if (row_start > col_start) return;
    // block内threads的大小: 在最后一个块内，如果小于最小的线程数，按照剩余线程数处理
    const int row_size = fminf(boxes_num - row_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);
    const int col_size = fminf(boxes_num - col_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);

    __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 7]; // 为同一个线程块中的线程申请共享内存
    
    // 将全局内存拷贝到共享内存上，boxes是一个一维的线性内存
    // 每个block有64个线程，所以复制64个候选框，每个候选框7个值，
    // 所以每个block分配的共享内存大小为64*7=448
    if (threadIdx.x < col_size) {
        block_boxes[threadIdx.x * 7 + 0] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 0];
        block_boxes[threadIdx.x * 7 + 1] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 1];
        block_boxes[threadIdx.x * 7 + 2] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 2];
        block_boxes[threadIdx.x * 7 + 3] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 3];
        block_boxes[threadIdx.x * 7 + 4] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 4];
        block_boxes[threadIdx.x * 7 + 5] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 5];
        block_boxes[threadIdx.x * 7 + 6] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 6];
    }
    __syncthreads(); // 线程同步，要求块中的线程必须等待所有线程都到达该点

    if (threadIdx.x < row_size) {
        // 每个线程虽然运行的是同一段代码，但看到的表示身份的threadIdx.x是不一样的，row_size值也不一样
        // 具体的说，有64*64个线程看到的threadIdx.x是一样的，因为总共有64*64个block，threadIdx.x的取值范围是0到63
        // 计算thread级的行号, 共4096个
        const int cur_box_idx = THREADS_PER_BLOCK_NMS * row_start + threadIdx.x;
        // 计算当前box的首地址
        const float *cur_box = boxes + cur_box_idx * 7;

        int i = 0;
        // t就是存放64个0、1的整数
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
          // 对角线上的block，对角线上的值不需要计算
          // 上三角与下三角是对称的，只需计算上三角
          start = threadIdx.x + 1;
        }
        // 当前box与块内的box计算iou
        for (i = start; i < col_size; i++) {
            if (iou_bev(cur_box, block_boxes + i * 7) > nms_overlap_thresh){
                // 每一个当前框都与其他框计算IOU，（其他框存放在共享内存，是一个复制品）
                // 本线程只负责计算第blockIdx.y×64+threadIdx.x号候选框（当前框）
                // 与blockIdx.x×64～blockIdx.x×64+63这64个候选框（其他框）之间的关系
                t |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS); // 64
        // mask总长为4096*64，表示4096行，64列的矩阵
        // 索引表示第cur_box_idx个候选框与第col_start×64到（col_start+1）×64-1之间这64个矩形框之间的重叠关系
        // 矩阵的索引即使[row * width + col]
        mask[cur_box_idx * col_blocks + col_start] = t;
        // 总共规划了（64,64,1）个block，每个block有（64,1,1）个线程，
        // 第（c，r，0）-（t,0,0)号线程负责计算的数据是：r×64+t号候选框与c×64～c×64+63号这64个候选框之间的重叠关系，
        // 并存进（r×64+t）×64+c这个整数里。
        // 当c，r遍历完64，t遍历完64，这64×64×64个线程就计算完了任意两个候选框之间的关系
    }
}


__device__ inline float iou_normal(float const * const a, float const * const b) {
    //params: a: [x, y, z, dx, dy, dz, heading]
    //params: b: [x, y, z, dx, dy, dz, heading]
    // 计算左上和右下坐标
    float left = fmaxf(a[0] - a[3] / 2, b[0] - b[3] / 2), right = fminf(a[0] + a[3] / 2, b[0] + b[3] / 2);
    float top = fmaxf(a[1] - a[4] / 2, b[1] - b[4] / 2), bottom = fminf(a[1] + a[4] / 2, b[1] + b[4] / 2);
    // 防止长宽小于0
    float width = fmaxf(right - left, 0.f), height = fmaxf(bottom - top, 0.f);
    // 以下同bev的iou计算相同
    float interS = width * height;
    float Sa = a[3] * a[4];
    float Sb = b[3] * b[4];
    return interS / fmaxf(Sa + Sb - interS, EPS);
}


__global__ void nms_normal_kernel(const int boxes_num, const float nms_overlap_thresh,
                           const float *boxes, unsigned long long *mask){
    //params: boxes (N, 7) [x, y, z, dx, dy, dz, heading]
    //params: mask (N, N/THREADS_PER_BLOCK_NMS)

    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    // if (row_start > col_start) return;

    const int row_size = fminf(boxes_num - row_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);
    const int col_size = fminf(boxes_num - col_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);

    __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 7];

    if (threadIdx.x < col_size) {
        block_boxes[threadIdx.x * 7 + 0] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 0];
        block_boxes[threadIdx.x * 7 + 1] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 1];
        block_boxes[threadIdx.x * 7 + 2] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 2];
        block_boxes[threadIdx.x * 7 + 3] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 3];
        block_boxes[threadIdx.x * 7 + 4] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 4];
        block_boxes[threadIdx.x * 7 + 5] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 5];
        block_boxes[threadIdx.x * 7 + 6] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 6];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = THREADS_PER_BLOCK_NMS * row_start + threadIdx.x;
        const float *cur_box = boxes + cur_box_idx * 7;

        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
          start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            if (iou_normal(cur_box, block_boxes + i * 7) > nms_overlap_thresh){
                t |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);
        mask[cur_box_idx * col_blocks + col_start] = t;
    }
}

/// @brief 计算box间的overlap
void boxesoverlapLauncher(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_overlap){

    dim3 blocks(DIVUP(num_b, THREADS_PER_BLOCK), DIVUP(num_a, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK); // (16, 16)

    boxes_overlap_kernel<<<blocks, threads>>>(num_a, boxes_a, num_b, boxes_b, ans_overlap);
#ifdef DEBUG
    // cudaDeviceSynchronize()函数保证device和host同步
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}

/// @brief 计算box间的iou
void boxesioubevLauncher(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_iou){

    dim3 blocks(DIVUP(num_b, THREADS_PER_BLOCK), DIVUP(num_a, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK); 

    boxes_iou_bev_kernel<<<blocks, threads>>>(num_a, boxes_a, num_b, boxes_b, ans_iou);
#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}

/// @brief 计算NMS
void nmsLauncher(const float *boxes, unsigned long long * mask, int boxes_num, float nms_overlap_thresh){
    dim3 blocks(DIVUP(boxes_num, THREADS_PER_BLOCK_NMS),
                DIVUP(boxes_num, THREADS_PER_BLOCK_NMS)); //（64, 64）
    dim3 threads(THREADS_PER_BLOCK_NMS); // 64
    // 启动核函数
    nms_kernel<<<blocks, threads>>>(boxes_num, nms_overlap_thresh, boxes, mask);
}

/// @brief 计算非负数面积的NMS
void nmsNormalLauncher(const float *boxes, unsigned long long * mask, int boxes_num, float nms_overlap_thresh){
    dim3 blocks(DIVUP(boxes_num, THREADS_PER_BLOCK_NMS),
                DIVUP(boxes_num, THREADS_PER_BLOCK_NMS));
    dim3 threads(THREADS_PER_BLOCK_NMS);
    nms_normal_kernel<<<blocks, threads>>>(boxes_num, nms_overlap_thresh, boxes, mask);
}
