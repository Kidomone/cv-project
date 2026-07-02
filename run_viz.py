import argparse
from src.utils.utils import visualize_random_prediction

def main():
    parser = argparse.ArgumentParser(description="Быстрый запуск визуализации предсказаний")
    parser.add_argument("--model", type=str, default="faster_rcnn", choices=["faster_rcnn", "ssd", "yolo"], help="Имя архитектуры")
    parser.add_argument("--weights", type=str, default="results/faster_rcnn_experiment/weights/best.pt", help="Путь к весам .pt")
    parser.add_argument("--config", type=str, default="data/raw/road-sign-detection-2/data.yaml", help="Путь к data.yaml датасета")
    args = parser.parse_args()

    print(f"Запуск визуализации для модели: {args.model}")
    print(f"Используем веса: {args.weights}")
    print(f"Используем конфиг данных: {args.config}")
    
    visualize_random_prediction(
        model_path=args.weights,
        dataset_yaml_path=args.config,
        model_name=args.model
    )

if __name__ == "__main__":
    main()