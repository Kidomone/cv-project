import argparse
import os
import yaml
from src.training.train import run_training

def main():
    parser = argparse.ArgumentParser(description="Пайплайн обучения 5 моделей для учебной практики")
    
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
        choices=['yolov8', 'yolo26', 'rt-detr', 'faster_rcnn', 'ssd'], 
        help="Имя модели для запуска обучения"
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.yaml', 
        help="Путь к файлу конфигурации"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Конфиг не найден: {args.config}")
        
    run_training(model_name=args.model, config_path=args.config)

if __name__ == '__main__':
    main()