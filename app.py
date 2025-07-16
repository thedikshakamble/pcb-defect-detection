import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import gradio as gr
from pathlib import Path

# ✅ Add yolov5 to Python path
sys.path.append('./yolov5')

# ✅ Import from YOLOv5
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# ✅ Load model
device = select_device('')
model = DetectMultiBackend('best.pt', device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = (640, 640)

# ✅ Detection function
def detect_pcb(img):
    img = img.convert("RGB")
    img = np.array(img)

    # Preprocess
    img_resized = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    img_resized /= 255.0
    img_resized = img_resized.to(device)

    pred = model(img_resized, augment=False)
    pred = non_max_suppression(pred)[0]

    # Draw boxes
    for *xyxy, conf, cls in pred:
        x1, y1, x2, y2 = map(int, xyxy)
        label = f'{names[int(cls)]} {conf:.2f}'
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        img = cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return Image.fromarray(img)

# ✅ Gradio UI
interface = gr.Interface(
    fn=detect_pcb,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="PCB Fault Detection",
    description="Upload an image of a PCB to detect faults using YOLOv5."
)

# ✅ For Render to expose port
port = int(os.environ.get("PORT", 7860))
interface.launch(server_name="0.0.0.0", server_port=port)
