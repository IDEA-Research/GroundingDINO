import typer
from groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm
import torchvision
import torch
import fiftyone as fo


def main(
        image_directory: str = 'test_grounding_dino',
        text_prompt: str = 'bus, car',
        box_threshold: float = 0.15, 
        text_threshold: float = 0.10,
        export_dataset: bool = False,
        view_dataset: bool = False,
        export_annotated_images: bool = True,
        weights_path : str = "groundingdino_swint_ogc.pth",
        config_path: str = "../../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        subsample: int = None,
    ):

    model = load_model(config_path, weights_path)
    
    dataset = fo.Dataset.from_images_dir(image_directory)

    samples = []

    if subsample is not None: 
        
        if subsample < len(dataset):
            dataset = dataset.take(subsample).clone()
    
    for sample in tqdm(dataset):

        image_source, image = load_image(sample.filepath)

        boxes, logits, phrases = predict(
            model=model, 
            image=image, 
            caption=text_prompt, 
            box_threshold=box_threshold, 
            text_threshold=text_threshold,
        )

        detections = [] 

        for box, logit, phrase in zip(boxes, logits, phrases):

            rel_box = torchvision.ops.box_convert(box, 'cxcywh', 'xywh')

            detections.append(
                fo.Detection(
                    label=phrase, 
                    bounding_box=rel_box,
                    confidence=logit,
            ))

        # Store detections in a field name of your choice
        sample["detections"] = fo.Detections(detections=detections)
        sample.save()

    # loads the voxel fiftyone UI ready for viewing the dataset.
    if view_dataset:
        session = fo.launch_app(dataset)
        session.wait()
        
    # exports COCO dataset ready for training
    if export_dataset:
        dataset.export(
            'coco_dataset',
            dataset_type=fo.types.COCODetectionDataset,
        )
        
    # saves bounding boxes plotted on the input images to disk
    if export_annotated_images:
        dataset.draw_labels(
            'images_with_bounding_boxes',
            label_fields=['detections']
        )


if __name__ == '__main__':
    typer.run(main)
