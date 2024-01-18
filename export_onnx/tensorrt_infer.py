import numpy as np
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    TrtRunner,
    engine_from_bytes,
)
from polygraphy.logger import G_LOGGER
import pathlib
from common import load_image, preprocess_prompt, to_numpy, postprocess_boxes, plot_boxes_to_image

def convert_int64_to_int32(tensor: np.ndarray) -> np.ndarray:
    if tensor.dtype == np.int64:
        return tensor.astype(np.int32)
    return tensor

def main():
    '''
    "samples", "input_ids", "token_type_ids", "text_token_mask", "text_self_attention_masks", "position_ids"
    torch.Size([1, 3, 800, 1440])
    torch.Size([1, 5])
    torch.Size([1, 5])
    torch.Size([1, 5])
    torch.Size([1, 5, 5])
    torch.Size([1, 5])
    '''
    import argparse
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--engine_path", "-m", type=pathlib.Path, required=True, help="path to config file")
    parser.add_argument("--image_path", "-i", type=pathlib.Path, required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=pathlib.Path, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")
    args = parser.parse_args()

    with open(args.engine_path, "rb") as engine_file:
        engine = engine_from_bytes(engine_file.read())
    print("engine loaded")
    runner = TrtRunner(engine.create_execution_context())
    print("execution context created")
    with runner:
        image_raw, image_tensor = load_image(args.image_path)
        image_tensor = image_tensor.unsqueeze(0)

        prompt = args.text_prompt
        text_encoder_type="bert-base-uncased"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
        inputs = preprocess_prompt(prompt=prompt, tokenizer=tokenizer)
        inputs["samples"] = image_tensor
        inputs = {name: to_numpy(value) for name, value in inputs.items()}
        # tensorrt doesn't support int64, so convert it to int32
        inputs = {name: convert_int64_to_int32(value) for name, value in inputs.items()}
        outputs = runner.infer(inputs)
        
        logits = outputs['logits']
        boxes = outputs['boxes']
        boxes_filt, pred_phrases = postprocess_boxes(logits, boxes, tokenizer, prompt,
                                                     box_threshold=args.box_threshold,
                                                     text_threshold=args.text_threshold,
                                                     token_spans=None)
        # visualize pred
        size = image_raw.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }
        # import ipdb; ipdb.set_trace()
        image_with_box = plot_boxes_to_image(image_raw, pred_dict)[0]
        image_with_box.save(args.output_dir / "pred_trt.jpg")

if __name__ == "__main__":
    main()