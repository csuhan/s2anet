// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <torch/extension.h>
#include <torch/types.h>


at::Tensor ROIAlignRotated_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio);

at::Tensor ROIAlignRotated_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio);

#if defined(WITH_CUDA) || defined(WITH_HIP)
at::Tensor ROIAlignRotated_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio);

at::Tensor ROIAlignRotated_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio);
#endif

// Interface for Python
inline at::Tensor ROIAlignRotated_forward(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio) {
  if (input.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return ROIAlignRotated_forward_cuda(
        input,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return ROIAlignRotated_forward_cpu(
      input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
}

inline at::Tensor ROIAlignRotated_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio) {
  if (grad.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return ROIAlignRotated_backward_cuda(
        grad,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        batch_size,
        channels,
        height,
        width,
        sampling_ratio);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return ROIAlignRotated_backward_cpu(
      grad,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      batch_size,
      channels,
      height,
      width,
      sampling_ratio);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "roi_align_rotated_forward",
      &ROIAlignRotated_forward,
      "Forward pass for Rotated ROI-Align Operator");
  m.def(
      "roi_align_rotated_backward",
      &ROIAlignRotated_backward,
      "Backward pass for Rotated ROI-Align Operator");
}

