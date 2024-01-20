import pathlib
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
import torch
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from typing import Union
from onnxsim import simplify, model_info
import sys

import logging
logger = logging.getLogger("groundingdino.export")
logger.setLevel(logging.INFO)
# add logger outputs to console.
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
fmt='%(asctime)s - %(filename)s:%(lineno)d - [%(levelname)s]: %(message)s'
sh.setFormatter(logging.Formatter(fmt))
logger.addHandler(sh)

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    model = model.to(device)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_fake_inputs(model, device: Union[str, torch.device]):
    H, W = 720, 1280
    image_f32 = torch.rand([1, 3, H, W], dtype=torch.float32, device=device)
    prompt = ["a cat."]
    tokenized = model.tokenizer(prompt, padding="longest", return_tensors="pt").to(device)
    from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
    (
        text_self_attention_masks,
        position_ids,
        cate_to_token_mask_list,
    ) = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, model.specical_tokens, model.tokenizer
    )
    input_ids = tokenized.input_ids
    token_type_ids = tokenized.token_type_ids
    text_token_mask = tokenized.attention_mask
    return image_f32, input_ids, token_type_ids, text_token_mask, text_self_attention_masks, position_ids

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model
    
    def forward(self, *args, **kwargs):
        return self._model.forward_nn(*args, **kwargs)

def export_model(model: torch.nn.Module, output_file: Union[str, pathlib.Path],
                 device: Union[str, torch.device], opset_version: int=17):
    from groundingdino.util.export_flag import ExportFlag
    fake_inputs = get_fake_inputs(model, device)
    # for item in fake_inputs:
    #     print(item.shape)
    # logger.info("try infer")
    # with torch.no_grad(), ExportFlag(True):
    #     model(*fake_inputs)
    logger.info("exporting model")
    with torch.no_grad(), ExportFlag(True):
        wrapped_model = ModelWrapper(model)
        torch.onnx.export(wrapped_model,
                          fake_inputs,
                          output_file,
                          opset_version=opset_version,
                          do_constant_folding=False,
                          input_names=["samples", "input_ids", "token_type_ids", "text_token_mask", "text_self_attention_masks", "position_ids"],
                          dynamic_axes={"input_ids": {1: "seq_len"},
                                        "token_type_ids": {1: "seq_len"},
                                        "text_token_mask": {1: "seq_len"},
                                        "text_self_attention_masks": {1: "seq_len", 2: "seq_len"},
                                        "position_ids": {1: "seq_len"},
                                        },
                          output_names=["logits", "boxes"]
                          )

from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

@parse_args('v','v','i','i','b')
def grid_sampler(g, input, grid, mode_enum, padding_mode_enum, align_corners):
    mode_str = ['bilinear', 'nearest', 'bicubic'][mode_enum]
    padding_str = ['zeros', 'border', 'reflection'][padding_mode_enum]
    return g.op('com.microsoft::GridSample',input,grid,mode_s=mode_str,padding_mode_s=padding_str,align_corners_i=align_corners)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Export Grounding DINO", add_help=True)
    parser.add_argument("--config_file", "-c", type=pathlib.Path, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=pathlib.Path, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=pathlib.Path, default="outputs", required=True, help="output directory"
    )
    parser.add_argument("--optimize", action="store_true", help="optimize exported onnx model when simplifying")
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    parser.add_argument("--opset", type=int, default=17, help="the opset to export onnx model")
    args = parser.parse_args()

    if args.opset < 17:
        register_custom_op_symbolic("::grid_sampler", grid_sampler, args.opset)

    if args.cpu_only:
        device = "cpu"
    elif not torch.cuda.is_available():
        device = "cpu"
        print("cpu-only is not configured, but no cuda available, use cpu")
    else:
        device = "cuda"
    device = torch.device(device)

    # cfg
    config_file: pathlib.Path = args.config_file  # change the path of the model config file
    checkpoint_path: pathlib.Path = args.checkpoint_path  # change the path of the model
    output_dir: pathlib.Path = args.output_dir

    # make dir
    output_dir.mkdir(exist_ok=True)
    # load model
    model = load_model(config_file, checkpoint_path, device=device)
    model_path = output_dir / "grounding_dino.onnx"
    export_model(model, model_path, device, opset_version=args.opset)
    print("exported model to {}".format(model_path))
    del model

    # simplify onnx model
    print("simplifying model")
    model_opt, success = simplify(str(model_path),
                                  perform_optimization=args.optimize,
                                  tensor_size_threshold="1KB")
    import onnx
    opt_model_path = output_dir / "grounding_dino_sim.onnx"
    onnx.save(model_opt, opt_model_path)
    print("saved simplified model to {0}".format(opt_model_path))

    if success:
        print("Finish! Here is the difference:")
        ori_model = onnx.load(model_path)
        model_info.print_simplifying_info(ori_model, model_opt)
    else:
        print(
            'Check failed. Please be careful to use the simplified model, or try specifying "skip-fuse-bn" or "skip-optimization" (see onnxsim.Simplify for details).'
        )
        print("Here is the difference after simplification:")
        ori_model = onnx.load(model_path)
        model_info.print_simplifying_info(ori_model, model_opt)
        sys.exit(1)