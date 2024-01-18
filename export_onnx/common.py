from PIL import Image, ImageDraw, ImageFont
import groundingdino.datasets.transforms as T
import torch
import numpy as np
import transformers

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([(1280, 720)], max_size=1440),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    raise ValueError("unahandled type {0}".format(type(tensor)))

from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
def preprocess_prompt(prompt: str, tokenizer: transformers.BertTokenizer):
    prompt = preprocess_caption(prompt)
    tokenized = tokenizer([prompt], padding="longest", return_tensors="pt").to('cpu')
    specical_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    #preprocess
    (
        text_self_attention_masks,
        position_ids,
        cate_to_token_mask_list,
    ) = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, specical_tokens, tokenizer
    )

    input_ids = tokenized.input_ids
    token_type_ids = tokenized.token_type_ids
    text_token_mask = tokenized.attention_mask

    inputs = {"input_ids": input_ids,
              "token_type_ids": token_type_ids,
              "text_token_mask": text_token_mask,
              "text_self_attention_masks": text_self_attention_masks,
              "position_ids": position_ids}
    return inputs

from groundingdino.util.utils import get_phrases_from_posmap
def postprocess_boxes(logits: np.ndarray, boxes: np.ndarray, tokenizer,
                      prompt: str, box_threshold: float, text_threshold: float, token_spans=None, with_logits=True, ):
    logits = torch.from_numpy(logits).sigmoid()[0] 
    boxes = torch.from_numpy(boxes)[0]
    # filter output
    if token_spans is None:
        logits_filt = logits.clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenized = tokenizer(prompt)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases
    return boxes_filt, pred_phrases