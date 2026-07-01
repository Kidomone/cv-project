import os
import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def visualize_random_prediction(model_path: str, dataset_yaml_path: str, output_dir: str = "results/visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Модель по пути {model_path} не найдена")
        return
    model = YOLO(model_path)
    
    base_dir = os.path.dirname(dataset_yaml_path)
    val_images_dir = os.path.join(base_dir, "valid/images")
    
    if not os.path.exists(val_images_dir):
        print(f"Папка валидации не найдена по пути: {val_images_dir}")
        return
        
    images = [f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print("В папке валидации нет изображений")
        return
        
    random_image_name = random.choice(images)
    image_path = os.path.join(val_images_dir, random_image_name)
    
    results = model(image_path, imgsz=416)
    
    annotated_frame = results[0].plot() 
    
    output_path = os.path.join(output_dir, f"pred_{random_image_name}")
    cv2.imwrite(output_path, annotated_frame)
    print(f"Визуализация предсказания сохранена в: {output_path}")
