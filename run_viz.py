import os
import argparse
from src.utils.utils import visualize_random_prediction

def main():
    parser = argparse.ArgumentParser(description="Запуск одиночной визуализации предсказаний")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["yolov8", "yolov26", "rt-detr", "faster_rcnn", "ssd"],
                        help="Имя модели для визуализации (yolov8, yolov26, rt-detr, faster_rcnn, ssd)")
    args = parser.parse_args()

    dataset_yaml = "data/raw/road-sign-detection-2/data.yaml" 
    
    model_path = f"results/{args.model}_experiment/weights/best.pt"

    print(f"Ищем веса по пути: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Ошибка: Файл весов не найден! Проверь, правильно ли указан путь.")
        return

    visualize_random_prediction(
        model_path=model_path,
        dataset_yaml_path=dataset_yaml,
        output_dir="results/visualizations",
        model_name=args.model
    )

if __name__ == "__main__":
    main()