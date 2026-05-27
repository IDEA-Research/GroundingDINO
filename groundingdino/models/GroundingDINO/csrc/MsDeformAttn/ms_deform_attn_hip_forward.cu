/*!
**************************************************************************************************
* Forward-only HIP implementation for GroundingDINO multi-scale deformable attention.
*
* This intentionally excludes the backward col2im kernels. Grounded-SAM style inference only needs
* ms_deform_attn_forward; training support should add a separate backward path.
**************************************************************************************************
*/

#include <algorithm>
#include <cmath>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "ms_deform_attn_hip.h"

namespace groundingdino {

namespace {

constexpr int kNumThreads = 256;

inline int get_blocks(const int n, const int num_threads) {
  return (n + num_threads - 1) / num_threads;
}

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear_forward(
    const scalar_t* bottom_data,
    const int height,
    const int width,
    const int num_heads,
    const int channels,
    const scalar_t h,
    const scalar_t w,
    const int head,
    const int channel) {
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh;
  const scalar_t hw = 1 - lw;

  const int w_stride = num_heads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = head * channels + channel;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    v1 = bottom_data[h_low_ptr_offset + w_low_ptr_offset + base_ptr];
  }

  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    v2 = bottom_data[h_low_ptr_offset + w_high_ptr_offset + base_ptr];
  }

  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    v3 = bottom_data[h_high_ptr_offset + w_low_ptr_offset + base_ptr];
  }

  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    v4 = bottom_data[h_high_ptr_offset + w_high_ptr_offset + base_ptr];
  }

  const scalar_t w1 = hh * hw;
  const scalar_t w2 = hh * lw;
  const scalar_t w3 = lh * hw;
  const scalar_t w4 = lh * lw;
  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename scalar_t>
__global__ void ms_deformable_im2col_forward_kernel(
    const int n,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* data_col) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < n;
       index += blockDim.x * gridDim.x) {
    int temp = index;
    const int channel = temp % channels;
    temp /= channels;
    const int sampling_index = temp;
    const int head = temp % num_heads;
    temp /= num_heads;
    const int query = temp % num_query;
    temp /= num_query;
    const int batch = temp;

    const int qid_stride = num_heads * channels;
    const int value_batch_offset = batch * spatial_size * qid_stride;
    int weight_ptr = sampling_index * num_levels * num_point;
    int loc_ptr = weight_ptr << 1;

    scalar_t out = 0;
    for (int level = 0; level < num_levels; ++level) {
      const int level_start_id = data_level_start_index[level];
      const int spatial_h_ptr = level << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const scalar_t* value_ptr =
          data_value + value_batch_offset + level_start_id * qid_stride;

      for (int point = 0; point < num_point; ++point) {
        const scalar_t loc_w = data_sampling_loc[loc_ptr];
        const scalar_t loc_h = data_sampling_loc[loc_ptr + 1];
        const scalar_t weight = data_attn_weight[weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          out += ms_deform_attn_im2col_bilinear_forward(
                     value_ptr, spatial_h, spatial_w, num_heads, channels,
                     h_im, w_im, head, channel) *
                 weight;
        }

        weight_ptr += 1;
        loc_ptr += 2;
      }
    }

    data_col[index] = out;
  }
}

template <typename scalar_t>
void ms_deformable_im2col_hip_forward(
    cudaStream_t stream,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* data_col) {
  const int num_kernels = batch_size * num_query * num_heads * channels;
  ms_deformable_im2col_forward_kernel<scalar_t>
      <<<get_blocks(num_kernels, kNumThreads), kNumThreads, 0, stream>>>(
          num_kernels, data_value, data_spatial_shapes, data_level_start_index,
          data_sampling_loc, data_attn_weight, batch_size, spatial_size,
          num_heads, channels, num_levels, num_query, num_point, data_col);
}

} // namespace

at::Tensor ms_deform_attn_hip_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step) {
  AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
  AT_ASSERTM(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
  AT_ASSERTM(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
  AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
  AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");

  AT_ASSERTM(value.is_cuda(), "value must be a CUDA/HIP tensor");
  AT_ASSERTM(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA/HIP tensor");
  AT_ASSERTM(level_start_index.is_cuda(), "level_start_index must be a CUDA/HIP tensor");
  AT_ASSERTM(sampling_loc.is_cuda(), "sampling_loc must be a CUDA/HIP tensor");
  AT_ASSERTM(attn_weight.is_cuda(), "attn_weight must be a CUDA/HIP tensor");

  const int batch = value.size(0);
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);
  const int num_levels = spatial_shapes.size(0);
  const int num_query = sampling_loc.size(1);
  const int num_point = sampling_loc.size(4);
  const int im2col_step_ = std::min(batch, im2col_step);

  AT_ASSERTM(batch % im2col_step_ == 0, "batch must divide im2col_step");

  auto output = at::zeros({batch, num_query, num_heads, channels}, value.options());
  auto output_n = output.view({batch / im2col_step_, im2col_step_, num_query, num_heads, channels});
  const int per_value_size = spatial_size * num_heads * channels;
  const int per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
  const int per_attn_weight_size = num_query * num_heads * num_levels * num_point;

  for (int n = 0; n < batch / im2col_step_; ++n) {
    auto columns = output_n.select(0, n);
    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_hip", ([&] {
      ms_deformable_im2col_hip_forward(
          at::cuda::getCurrentCUDAStream(),
          value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
          spatial_shapes.data_ptr<int64_t>(),
          level_start_index.data_ptr<int64_t>(),
          sampling_loc.data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
          attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
          im2col_step_, spatial_size, num_heads, channels, num_levels, num_query, num_point,
          columns.data_ptr<scalar_t>());
    }));
  }

  return output.view({batch, num_query, num_heads * channels});
}

} // namespace groundingdino
