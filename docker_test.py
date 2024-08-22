from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch
import cv2

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
model = model.to('cuda:0')
print(torch.cuda.is_available())
print('DONE!')
