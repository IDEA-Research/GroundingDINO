python convert_to_yolo_format.py \
    --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint_path weights/groundingdino_swinb_cogcoor.pth \
    --input medSample/ \
    --text_prompt "screen . monitor . display ." \
    --output_dir annotations_yolo


python convert_to_yolo_format.py \
    --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --checkpoint_path /home/aic/.cache/huggingface/hub/models--ShilongLiu--GroundingDINO/snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/groundingdino_swint_ogc.pth \
    --input medSample/ \
    --text_prompt "screen" \
    --output_dir annotations_yolo \
    --keep_all

python convert_to_yolo_format.py \
    --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --checkpoint_path /home/aic/.cache/huggingface/hub/models--ShilongLiu--GroundingDINO/snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/groundingdino_swint_ogc.pth \
    --input medSample/ \
    --text_prompt "screen" \
    --output_dir annotations_yolo \
    --keep_all


python convert_to_yolo_format.py \
    --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --checkpoint_path /home/aic/.cache/huggingface/hub/models--ShilongLiu--GroundingDINO/snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/groundingdino_swint_ogc.pth \
    --input medSample/ \
    --text_prompt "screen" \
    --output_dir annotations_yolo \
    --iou_threshold 0.3 \
    --keep_all