from typing import Tuple, List, Any

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
import torchvision
from torchvision.ops import box_convert
import torchvision.transforms.functional as F
import bisect

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap

# ----------------------------------------------------------------------------------------------------------------------
# OLD API
# ----------------------------------------------------------------------------------------------------------------------


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def predict(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
        
        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]

    return boxes, logits.max(dim=1)[0], phrases


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates.
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


# ----------------------------------------------------------------------------------------------------------------------
# NEW API
# ----------------------------------------------------------------------------------------------------------------------


class Model:

    def __init__(
        self,
        model_config_path: str,
        model_checkpoint_path: str,
        device: str = "cuda"
    ):
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device
        ).to(device)
        self.device = device

    def predict_with_caption(
        self,
        image: np.ndarray,
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> Tuple[sv.Detections, List[str]]:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections, labels = model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        """
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold, 
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        return detections, phrases

    def predict_with_classes(
        self,
        image: np.ndarray,
        classes: List[str],
        box_threshold: float,
        text_threshold: float
    ) -> sv.Detections:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections = model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )


        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        """
        caption = ". ".join(classes)
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        class_id = Model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        return detections

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: torch.Tensor,
            logits: torch.Tensor
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)

    @staticmethod
    def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            for class_ in classes:
                if class_ in phrase:
                    class_ids.append(classes.index(class_))
                    break
            else:
                class_ids.append(None)
        return np.array(class_ids)


#==============================================================================


class BatchedModel(object):

#=====================================================

    def __init__(
        self,
        model_config_path: str,
        model_checkpoint_path: str,
        device: str = "cuda",
        dtype: str = "float32",
        compile: bool = False
    ) -> NotImplementedError:

        self._device = device
        self._dtype = getattr(torch, dtype)
        self._model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path
        ).to(device=self._device).to(dtype=self._dtype)

        # Compile model if necessary
        if compile:
            self._model = torch.compile(self._model)

#=====================================================

    @staticmethod
    def preprocess_image(
        image_batch: torch.Tensor
    ) -> torch.Tensor:

        # Preprocessing friendly with batches

        image_batch = F.resize(image_batch, [800], antialias=True)
        image_batch = F.normalize(image_batch, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   
        return image_batch

#=====================================================

    @classmethod
    def post_process_result(
            cls,
            boxes_cxcywh: torch.Tensor,
            logits: torch.Tensor,
            nms_threshold: float,
            source_size: Tuple[int, int],
            phrases: List[str],
            text_prompts: List[str]
    ):

        bbox_batch, conf_batch, class_id_batch = [], [], []
        source_h, source_w = source_size
        for bbox_cxcywh, conf, phrase, text_prompt in zip(boxes_cxcywh, logits, phrases, text_prompts):
            bbox_cxcywh *= torch.Tensor([source_w, source_h, source_w, source_h])
            bbox_xyxy = box_convert(boxes=bbox_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")

            # Perform NMS
            nms_idx = torchvision.ops.nms(bbox_xyxy.float(), conf.float(), nms_threshold).numpy().tolist()
            class_id = cls.phrases2classes(phrases=phrase, classes=text_prompt)

            bbox_batch.append(bbox_xyxy[nms_idx])
            conf_batch.append(conf[nms_idx])
            class_id_batch.append(class_id[nms_idx])

        return bbox_batch, conf_batch, class_id_batch

#=====================================================

    def _batched_predict(
        self,
        image_batch,
        text_prompts,
        box_threshold,
        text_threshold
    ):
        # Predict refactored to work with batches
        captions = [preprocess_caption(caption) for caption in text_prompts]

        outputs = self._model(image_batch, captions=captions)

        prediction_logits = outputs["pred_logits"].cpu().sigmoid()  # prediction_logits.shape = (bsz，nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()  # prediction_boxes.shape = (bsz, nq, 4)

        logits_res = []
        boxs_res = []
        phrases_list = []
        tokenizer = self._model.tokenizer
        for ub_logits, ub_boxes, ub_captions in zip(prediction_logits, prediction_boxes, captions):
            mask = ub_logits.max(dim=1)[0] > box_threshold
            logits = ub_logits[mask]  # logits.shape = (n, 256)
            boxes = ub_boxes[mask]  # boxes.shape = (n, 4)
            logits_res.append(logits.max(dim=1)[0])
            boxs_res.append(boxes)

            tokenized = tokenizer(ub_captions)
            phrases = [
                get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
                for logit
                in logits
            ]
            phrases_list.append(phrases)

        return boxs_res, logits_res, phrases_list

    def predict(
        self,
        image_batch: torch.Tensor,
        text_prompts: List[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.3,
        nms_threshold: float = 0.5
    ):

        # Move to device and type just in case
        image_batch = image_batch.to(device=self._device).to(dtype=self._dtype)
        source_h, source_w = image_batch.shape[-2:]

        if any(isinstance(i, list) for i in text_prompts):
            captions = [". ".join(text_prompt) for text_prompt in text_prompts]
        else:
            captions = [". ".join(text_prompts)]
            text_prompts = [text_prompts]

        # Extend caption to batch
        if len(captions) == 1:
            captions *= image_batch.shape[0]
        if len(text_prompts) == 1:
            text_prompts *= image_batch.shape[0]

        # Preprocess, inference and postprocess
        processed_image = self.preprocess_image(image_batch)
        bboxes, logits, phrases = self._batched_predict(
            processed_image, 
            captions, 
            box_threshold, 
            text_threshold
        )
        bbox_batch, conf_batch, class_id_batch = self.post_process_result(
            bboxes, 
            logits, 
            nms_threshold,
            (source_h, source_w),
            phrases,
            text_prompts
        )

        return bbox_batch, conf_batch, class_id_batch

    @staticmethod
    def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            for class_ in classes:
                if class_.lower() in phrase.lower():
                    class_ids.append(classes.index(class_))
                    break
            else:
                class_ids.append(None)
        return np.array(class_ids)


    def __call__(
        self,
        *args,
        **kwargs
    ) -> Any:
        return self.predict(*args, **kwargs)
