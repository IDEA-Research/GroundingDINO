// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include "MsDeformAttn/ms_deform_attn.h"

#include <torch/library.h>

namespace groundingdino {

#ifdef WITH_CUDA
extern int get_cudart_version();
#endif

std::string get_cuda_version() {
#ifdef WITH_CUDA
  std::ostringstream oss;

  // copied from
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/CUDAHooks.cpp#L231
  auto printCudaStyleVersion = [&](int v) {
    oss << (v / 1000) << "." << (v / 10 % 100);
    if (v % 10 != 0) {
      oss << "." << (v % 10);
    }
  };
  printCudaStyleVersion(get_cudart_version());
  return oss.str();
#else
  return std::string("not available");
#endif
}

// similar to
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Version.cpp
std::string get_compiler_version() {
  std::ostringstream ss;
#if defined(__GNUC__)
#ifndef __clang__
  { ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__; }
#endif
#endif

#if defined(__clang_major__)
  {
    ss << "clang " << __clang_major__ << "." << __clang_minor__ << "."
       << __clang_patchlevel__;
  }
#endif

#if defined(_MSC_VER)
  { ss << "MSVC " << _MSC_FULL_VER; }
#endif
  return ss.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
}

at::Tensor ms_deform_attn_forward_dispatch(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    int64_t im2col_step) {
  return ms_deform_attn_forward(
      value,
      spatial_shapes,
      level_start_index,
      sampling_loc,
      attn_weight,
      static_cast<int>(im2col_step));
}

} // namespace groundingdino

TORCH_LIBRARY(groundingdino, m) {
  m.def("ms_deform_attn_forward(Tensor value, Tensor value_spatial_shapes, "
        "Tensor value_level_start_index, Tensor sampling_locations, "
        "Tensor attention_weights, int im2col_step) -> Tensor");
}

TORCH_LIBRARY_IMPL(groundingdino, CUDA, m) {
  m.impl("ms_deform_attn_forward", &groundingdino::ms_deform_attn_forward_dispatch);
}