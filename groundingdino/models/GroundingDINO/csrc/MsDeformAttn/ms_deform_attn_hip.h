/*!
**************************************************************************************************
* Forward-only HIP path for GroundingDINO multi-scale deformable attention.
**************************************************************************************************
*/

#pragma once
#include <torch/extension.h>

namespace groundingdino {

at::Tensor ms_deform_attn_hip_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step);

} // namespace groundingdino
