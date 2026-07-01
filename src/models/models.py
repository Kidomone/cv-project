# src/models/models.py
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, SSD300_VGG16_Weights
from ultralytics import YOLO, RTDETR

def get_model(model_name: str, num_classes: int):

    name = model_name.lower()
    
    if name == 'yolov8':
        print("Инициализация YOLOv8...")
        return YOLO('yolov8n.pt')
        
    elif name == 'yolo26':
        print("Инициализация YOLO26...")
        return YOLO('yolo26n.pt')
        
    elif name == 'rt-detr':
        print("Инициализация RT-DETR...")
        return RTDETR('rtdetr-l.pt')
        
    elif name == 'faster_rcnn':
        print("Инициализация Faster R-CNN (ResNet50)...")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes + 1)
        return model
        
    elif name == 'ssd':
        print("Инициализация SSD (VGG16)...")
        weights = SSD300_VGG16_Weights.DEFAULT
        model = torchvision.models.detection.ssd300_vgg16(weights=weights)
        return model
        
    else:
        raise ValueError(f"Модель '{model_name}' не поддерживается. Выберите из: yolov8, yolo26, rt-detr, faster_rcnn, ssd")