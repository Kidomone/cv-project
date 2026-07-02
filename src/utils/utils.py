import os
import random
import cv2
import yaml
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def visualize_random_prediction(model_path: str, dataset_yaml_path: str, output_dir: str = "results/visualizations", model_name: str = "faster_rcnn"):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Модель по пути {model_path} не найдена")
        return

    with open(dataset_yaml_path, 'r') as f:
        info = yaml.safe_load(f)
    
    names_data = info.get('names', [])
    if isinstance(names_data, dict):
        class_names = list(names_data.values())
    else:
        class_names = list(names_data)

    base_dir = os.path.dirname(os.path.abspath(dataset_yaml_path))
    split_dir = info.get('val', 'valid/images')
    clean_split_dir = split_dir.replace('../', '').replace('./', '')
    val_images_dir = os.path.abspath(os.path.join(base_dir, clean_split_dir))
    
    if not os.path.exists(val_images_dir):
        print(f"Папка валидации не найдена по пути: {val_images_dir}")
        return
        
    images = [f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print("В папке валидации нет изображений")
        return
        
    random_image_name = random.choice(images)
    image_path = os.path.join(val_images_dir, random_image_name)
    output_path = os.path.join(output_dir, f"pred_{random_image_name}")

    if 'yolo' in model_name.lower():
        from ultralytics import YOLO
        model = YOLO(model_path)
        results = model(image_path, imgsz=416)
        annotated_frame = results[0].plot() 
        cv2.imwrite(output_path, annotated_frame)
        print(f"Визуализация предсказания YOLO сохранена в: {output_path}")
        return

    checkpoint = torch.load(model_path, map_location='cpu')
    num_classes = int(info.get('nc', 4)) + 1
    
    if 'faster_rcnn' in model_name.lower():
        model = fasterrcnn_resnet50_fpn(num_classes=num_classes, pretrained=False)
    else:
        from torchvision.models.detection import ssd300_vgg16
        model = ssd300_vgg16(num_classes=num_classes, pretrained=False)
        
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint.state_dict())
        
    model.eval()
    
    orig_image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    
    transform = T.Compose([T.ToTensor()])
    input_tensor = transform(image_rgb).unsqueeze(0)
    
    with torch.no_grad():
        predictions = model(input_tensor)[0]
        
    boxes = predictions['boxes'].numpy()
    labels = predictions['labels'].numpy()
    scores = predictions['scores'].numpy()
    
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            xmin, ymin, xmax, ymax = map(int, box)
            cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            class_idx = label - 1
            class_name = class_names[class_idx] if 0 <= class_idx < len(class_names) else f"Class {label}"
            text = f"{class_name} {score:.2f}"
            
            cv2.putText(orig_image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    cv2.imwrite(output_path, orig_image)
    print(f"Визуализация предсказания PyTorch ({model_name}) сохранена в: {output_path}")