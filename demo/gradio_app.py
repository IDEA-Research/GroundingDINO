import argparse
import cv2
from PIL import Image
import numpy as np

import warnings

import torch

# prepare the environment
# os.system("python setup.py build develop --user")
# os.system("pip install packaging==21.3")
# os.system("pip install gradio")


warnings.filterwarnings("ignore")

import gradio as gr

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict, load_model
import groundingdino.datasets.transforms as T


def image_transform_grounding(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return init_image, image

def image_transform_grounding_for_vis(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return image

config_file = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "weights/groundingdino_swinb_cogcoor.pth"

model = load_model(config_file, ckpt_filenmae, device='cuda')

def run_grounding(input_image, grounding_caption, box_threshold, text_threshold):
    init_image = input_image.convert("RGB")
    _, image_tensor = image_transform_grounding(init_image)
    image_pil: Image = image_transform_grounding_for_vis(init_image)

    # run grounidng
    boxes, logits, phrases = predict(model, image_tensor, grounding_caption, box_threshold, text_threshold, device='cuda')
    annotated_frame = annotate(image_source=np.asarray(image_pil), boxes=boxes, logits=logits, phrases=phrases)
    image_with_box = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))


    return image_with_box

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    args = parser.parse_args()

    block = gr.Blocks().queue()
    with block:
        gr.Markdown("# [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)")
        gr.Markdown("### Open-World Detection with Grounding DINO")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label='upload', type="pil")
                grounding_caption = gr.Textbox(label="Detection Prompt")
                run_button = gr.Button(value="Run")
                with gr.Accordion("Advanced options", open=False):
                    box_threshold = gr.Slider(
                        label="Box Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )
                    text_threshold = gr.Slider(
                        label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )

            with gr.Column():
                gallery = gr.components.Image(
                    label="grounding results",
                    type="pil"
                )

        run_button.click(fn=run_grounding, inputs=[
                        input_image, grounding_caption, box_threshold, text_threshold], outputs=[gallery])


    block.launch(server_name='0.0.0.0', server_port=7579, debug=args.debug, share=args.share)
