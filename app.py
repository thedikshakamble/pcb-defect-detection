import torch
from pathlib import Path
from PIL import Image
import gradio as gr
import sys

# Clone YOLOv5 repo (assumes 'yolov5' folder already exists in repo)
sys.path.append('./yolov5')
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.datasets import LoadImages
from utils.torch_utils import select_device
import numpy as np

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
    img_resized /= 255.0  # normalize
    img_resized = img_resized.to(device)

    pred = model(img_resized, augment=False)
    pred = non_max_suppression(pred)[0]

    # Draw boxes (simple version)
    for *xyxy, conf, cls in pred:
        x1, y1, x2, y2 = map(int, xyxy)
        label = f'{names[int(cls)]} {conf:.2f}'
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        img = cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    return Image.fromarray(img)

# Gradio UI
interface = gr.Interface(
    fn=detect_pcb,
    inputs=gr.Image(type="pil", label="Upload PCB Image"),
    outputs=gr.Image(type="pil", label="Detected Defects"),
    title="PCB Defect Detector",
    description="Upload a PCB image to detect defects using YOLOv5."
)

interface.launch()
