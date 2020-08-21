// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once
#include <torch/extension.h>
#include <torch/types.h>

at::Tensor nms_rotated_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const float iou_threshold);

#ifdef WITH_CUDA
at::Tensor nms_rotated_cuda(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const float iou_threshold);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
inline at::Tensor nms_rotated(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const float iou_threshold) {
  assert(dets.device().is_cuda() == scores.device().is_cuda());
  if (dets.device().is_cuda()) {
#ifdef WITH_CUDA
    return nms_rotated_cuda(dets, scores, labels, iou_threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  return nms_rotated_cpu(dets, scores, labels, iou_threshold);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ml_nms_rotated", &nms_rotated, "multi label NMS for rotated boxes");

}
