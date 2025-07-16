import gradio as gr
import numpy as np
from PIL import Image
import torch
import sys

# Lazy load YOLOv5 only when needed
def detect_pcb(img):
    import cv2
    sys.path.append('./yolov5')
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression
    from utils.torch_utils import select_device

    # Load model
    device = select_device('')
    model = DetectMultiBackend('yolov5s.pt', device=device)  # Use lightest model
    names = model.names

    img = img.convert("RGB")
    img = np.array(img)

    # Format image
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    pred = model(img_tensor, augment=False)
    pred = non_max_suppression(pred)[0]

    # Draw boxes
    for *xyxy, conf, cls in pred:
        x1, y1, x2, y2 = map(int, xyxy)
        label = f'{names[int(cls)]} {conf:.2f}'
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return Image.fromarray(img)

# Gradio UI
demo = gr.Interface(
    fn=detect_pcb,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="PCB Defect Detector (YOLOv5s)",
)

demo.launch()
