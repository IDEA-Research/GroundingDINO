import pdb
from groundingdino.util.inference import load_model, load_image, predict, annotate
import onnxruntime as ort
import pathlib
import torch
import time
from common import load_image, plot_boxes_to_image, preprocess_prompt, to_numpy, postprocess_boxes
import numpy as np
from transformers import AutoTokenizer


def get_onnx_grounding_output(session: ort.InferenceSession, image, prompt, tokenizer, box_threshold, text_threshold=None, with_logits=True, token_spans=None):
    image = image.unsqueeze(0).numpy()

    inputs = preprocess_prompt(prompt=prompt, tokenizer=tokenizer)
    inputs["samples"] = image
    inputs = {name: to_numpy(value) for name, value in inputs.items()}
    outputs = session.run(None, inputs)
    # import ipdb ; ipdb.set_trace()

    logits = outputs[0]
    boxes = outputs[1]
    return postprocess_boxes(logits, boxes, tokenizer, prompt, box_threshold, text_threshold, token_spans)

def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', "CPUExecutionProvider"])
    return session

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Infer Grounding DINO onnx model.", add_help=True)
    parser.add_argument("--onnx_path", "-m", type=pathlib.Path, required=True, help="path to config file")
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
    
    session = load_onnx_model(args.onnx_path)

    image_raw, image_tensor = load_image(args.image_path)
    text_encoder_type="bert-base-uncased"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    boxes_filt, pred_phrases = get_onnx_grounding_output(session, image_tensor, args.text_prompt, tokenizer=tokenizer, box_threshold=args.box_threshold, text_threshold=args.text_threshold)

    # visualize pred
    size = image_raw.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    # import ipdb; ipdb.set_trace()
    image_with_box = plot_boxes_to_image(image_raw, pred_dict)[0]
    image_with_box.save(args.output_dir / "pred_onnx.jpg")