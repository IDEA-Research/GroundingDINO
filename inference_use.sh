CUDA_VISIBLE_DEVICES=0 python demo/inference_on_a_image.py \
-c groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p /home/aic/.cache/huggingface/hub/models--ShilongLiu--GroundingDINO/snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/groundingdino_swint_ogc.pth \
-i med_frame/frame_000008.jpg \
-o output/ \
-t "screen"


CUDA_VISIBLE_DEVICES=0 python inference_screen_crop.py \
-c groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p /home/aic/.cache/huggingface/hub/models--ShilongLiu--GroundingDINO/snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/groundingdino_swint_ogc.pth \
-i med_frame/frame_000008.jpg \
-o output/ \
-t "screen" \
--iou_threshold 0.3 \
--bezel_expansion 0.05


python screen_crop_demo.py -i med_frame/frame_000008.jpg -o demo_output