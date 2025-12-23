import gradio as gr
from ultralytics import YOLO
import os

# Поиск модели, если путь не указан

model_paths = [
    "runs/detect/train_optimized/weights/best.pt",
    "runs/detect/train_optimized/weights/last.pt",
    "runs/detect/train5/weights/best.pt",
    "runs/detect/train5/weights/last.pt",
    "runs/detect/train/weights/best.pt",
    "runs/detect/train/weights/last.pt"
]
    
model = None
for path in model_paths:
    if os.path.exists(path):
        model = path
        break

def infer(img):
    results = model(img)
    return results[0].plot()  # в Gradio выведется картинка с боксами

interface = gr.Interface(
    fn=infer,
    inputs=gr.Image(type="numpy"),
    outputs="image"
)
interface.launch()
