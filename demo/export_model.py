import pathlib
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
import torch
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from typing import Union

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

def export_model(model: torch.nn.Module, output_file: Union[str, pathlib.Path], device: Union[str, torch.device]):
    from groundingdino.util.export_flag import ExportFlag
    fake_inputs = get_fake_inputs(model, device)
    # logger.info("try infer")
    # with torch.no_grad(), ExportFlag(True):
    #     model(*fake_inputs)
    logger.info("exporting model")
    with torch.no_grad(), ExportFlag(True):
        wrapped_model = ModelWrapper(model)
        torch.onnx.export(wrapped_model, fake_inputs, output_file, opset_version=17)

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
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    args = parser.parse_args()

    device = "cuda" if not args.cpu_only else "cpu"
    device = torch.device(device)

    # cfg
    config_file: pathlib.Path = args.config_file  # change the path of the model config file
    checkpoint_path: pathlib.Path = args.checkpoint_path  # change the path of the model
    output_dir: pathlib.Path = args.output_dir

    # make dir
    output_dir.mkdir(exist_ok=True)
    # load model
    model = load_model(config_file, checkpoint_path, device=device)
    export_model(model, output_dir / "grouding_dino.onnx", device)