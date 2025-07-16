import torch
from pathlib import Path
from PIL import Image
import gradio as gr
import sys
import os
import numpy as np
import cv2

# Add YOLOv5 repo to path (assumes yolov5 folder is in same directory)
sys.path.append('./yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.datasets import LoadImages
from utils.torch_utils import select_device

# Load model
device = select_device('')
model = DetectMultiBackend('best.pt', device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = (640, 640)

# Inference function
def detect_pcb(img):
    img = img.convert("RGB")
    img = np.array(img)

    # Convert to batch and channel-first format
    img_resized = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    img_resized /= 255.0
    img_resized = img_resized.to(device)

    pred = model(img_resized, augment=False)
    pred = non_max_suppression(pred)[0]

    for *xyxy, conf, cls in pred:
        x1, y1, x2, y2 = map(int, xyxy)
        label = f'{names[int(cls)]} {conf:.2f}'
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2.putText(img, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return Image.fromarray(img)

# Gradio interface
interface = gr.Interface(
    fn=detect_pcb,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="PCB Fault Detection",
    description="Upload an image of a PCB to detect faults using YOLOv5."
)

# Run app with correct host and port for Render
port = int(os.environ.get("PORT", 7860))
interface.launch(server_name="0.0.0.0", server_port=port)
