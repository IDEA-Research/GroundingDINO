这里记录了从导出onnx到转tensorrt的坑和解决方案
# 导出onnx
## 模型修改
为了导出onnx模型，需要进行如下修改：
1. `MultiScaleDeformableAttnFunction`无法被直接trace，且没有对应的现成onnx、tensorrt算子，所以需要原生torch API的版本，即`multi_scale_deformable_attn_pytorch`，为了解决这个问题我加了个ExportFlag (`groundingdino/util/export_flag.py`)，用来标记目前是否在进行模型导出。然后在MultiScaleDeformableAttention.forward中（groundingdino/models/GroundingDINO/transformer_vanilla.py:331），如果发现是在进行模型导出就用 `multi_scale_deformable_attn_pytorch`。

2. 导出报错：
    ```
    ======================= 0 NONE 0 NOTE 0 WARNING 1 ERROR ========================
    ERROR: missing-standard-symbolic-function
    =========================================
    Exporting the operator 'aten::__ior_' to ONNX opset version 17 is not supported. Please feel free to request support or submit a pull request on PyTorch GitHub: https://github.com/pytorch/pytorch/issues.
    None
    <Set verbose=True to see more details>
    ```
    这个问题是因为`generate_masks_with_special_tokens_and_transfer_map`函数（`groundingdino/models/GroundingDINO/bertwarper.py:237`）和`generate_masks_with_special_tokens`函数（`groundingdino/models/GroundingDINO/bertwarper.py:193`），有一个inplace or运算`special_tokens_mask |= input_ids == special_token`，这个没法导出。然而实际上这个操作inplace是没必要的，直接改成pure函数即可，需要改成 `special_tokens_mask = torch.logical_or(special_tokens_mask, input_ids == special_token)`。

3. 需要注意模型的输入是图像+查询目标的描述文本（target，字符串类型），其中target会传给tokenizer，进行分词和token id查询操作。如果直接用GroundingDino.forward函数（`groundingdino/models/GroundingDINO/groundingdino.py:227`）进行trace，那么tokenizer查询出来的input_ids会被直接作为constant放到模型里，显然不是我们的期望。因此我将forward拆分成了tokenizer部分和`forward_nn`（`groundingdino/models/GroundingDINO/groundingdino.py:288`）部分，其中tokenizer部分执行分词操作，分词出的input_ids、attention mask等tensor作为参数传递给forward_nn函数，导出时用forward_nn来导出。

4. shape计算的修改：模型中有很多 spatial_shapes 相关的shape计算，而这个tensor是放在GPU上的，进行shape计算的时候需要把tensor从显卡复制回CPU，例如：
    ```
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)
    ```
    而这个操作是implicitly执行的，导出的时候会出问题，因此很多地方都做了这样的修改：`spatial_shapes => spatial_shapes.cpu()`
    顺便说一句，即使不导出的时候这种写法功能上没问题，但它会引入无用的D2H复制，降低代码在torch里的训练、推理性能，可以考虑把这个tensor在cpu上也放一份。

## 导出的注意事项
1. 模型里边有很多跟图像shape相关的分支，程序会根据shape不同走不同分支。而这些操作无法trace，从日志中warning也能看出来，所有的条件语句都会在trace的时候固定下来。这就导致如果部署后输入shape变了，模型不会按预期的来执行，因此目前都是固定图像shape来导出的。
2. 注意输入prompt的input_ids长度是会变的，所以这个需要动态shape。导出onnx的时候需要指定对应维度为dynamic（export_onnx/export_model.py:74），转tensorrt的时候要指定shape范围（export_onnx/convert_tensorrt.py:33-38），如果最终token的长度跟这儿写的不一致要对应进行修改。

# 依赖
```
Tensorrt 8.6：https://developer.nvidia.com/nvidia-tensorrt-8x-download
```
python依赖就直接 `pip install -r requirements-export.txt`

# 使用方法
```
cd export_onnx
python export_model.py -c ../groundingdino/config/GroundingDINO_SwinT_OGC_export.py -p ../weights/groundingdino_swint_ogc.pth --output_dir ./ --optimize
python convert_tensorrt.py -m ./grounding_dino_sim.onnx -o grounding_dino.trtengine 
python tensorrt_infer.py --engine_path ./grounding_dino.trtengine  -i ../test/cat.jpg -t 'cat' --output_dir ../test/result
```
