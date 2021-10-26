#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

// 把写好的头文件包含进来
#include "iou3d_cpu.h"
#include "iou3d_nms.h"


// 使用PYBIND11_MODULE进行函数绑定，在<torch/extension.h>中被包含
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	// 第一参数表示在python中调用时的函数名
	// 第二个参数是.h文件中定义的函数
	// 第三个参数是python中调用help所产生的提示
	m.def("boxes_overlap_bev_gpu", &boxes_overlap_bev_gpu, "oriented boxes overlap");
	m.def("boxes_iou_bev_gpu", &boxes_iou_bev_gpu, "oriented boxes iou");
	m.def("nms_gpu", &nms_gpu, "oriented nms gpu");
	m.def("nms_normal_gpu", &nms_normal_gpu, "nms gpu");
	m.def("boxes_iou_bev_cpu", &boxes_iou_bev_cpu, "oriented boxes iou");
}
