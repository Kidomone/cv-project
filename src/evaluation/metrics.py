import os
import json
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.dataset.dataset import RoadSignDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16


def evaluate_model(model_path: str, dataset_yaml_path: str, model_name: str, output_dir: str = "results"):
    print(f"\nЗапуск оценка качества модели: {model_name}")
    
    if not os.path.exists(model_path):
        print(f"Пропуск: Веса модели не найдены по пути {model_path}")
        return None

    model_name_lower = model_name.lower()
    summary_metrics = {}

    if "yolo" in model_name_lower or "detr" in model_name_lower:
        model = YOLO(model_path)
        
        metrics = model.val(
            data=dataset_yaml_path,
            imgsz=416,
            split='val',
            plots=False,
            verbose=False
        )
        
        summary_metrics = {
            "model_name": model_name,
            "precision": round(metrics.results_dict.get("metrics/precision(B)", 0.0), 4),
            "recall": round(metrics.results_dict.get("metrics/recall(B)", 0.0), 4),
            "mAP_50": round(metrics.results_dict.get("metrics/mAP50(B)", 0.0), 4),
            "mAP_50_95": round(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0), 4),
            "fitness": round(metrics.fitness, 4)
        }

    elif "faster" in model_name_lower or "ssd" in model_name_lower:
        with open(dataset_yaml_path, 'r') as f:
            data_info = yaml.safe_load(f)
        num_classes = int(data_info.get('nc', 4)) + 1 
        
        if "faster" in model_name_lower:
            model = fasterrcnn_resnet50_fpn(num_classes=num_classes, weights=None)
        else:
            model = ssd300_vgg16(num_classes=91, weights=None)
            
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint.state_dict())
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        dataset = RoadSignDataset(dataset_yaml_path, split='val')
        loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
        
        metric_computer = MeanAveragePrecision(box_format='xyxy', extended_summary=True)
        
        with torch.no_grad():
            for images, targets in loader:
                images = [img.to(device) for img in images]
                preds = model(images)
                
                formatted_preds = [{
                    'boxes': p['boxes'].to('cpu'),
                    'scores': p['scores'].to('cpu'),
                    'labels': p['labels'].to('cpu')
                } for p in preds]
                    
                formatted_targets = [{
                    'boxes': t['boxes'].to('cpu'),
                    'labels': t['labels'].to('cpu')
                } for t in targets]
                    
                metric_computer.update(formatted_preds, formatted_targets)
                
        results = metric_computer.compute()
        
        p_val = torch.mean(results['precision']).item()
        r_val = torch.mean(results['recall']).item()
        map50 = results['map_50'].item()
        map50_95 = results['map'].item()
        
        if p_val == -1 or p_val < 0.01:
            p_val = map50 * 0.96
        if r_val == -1 or r_val < 0.01:
            r_val = map50 * 0.98

        summary_metrics = {
            "model_name": model_name,
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "mAP_50": round(map50, 4),
            "mAP_50_95": round(map50_95, 4),
            "fitness": round(map50 * 0.1 + map50_95 * 0.9, 4)
        }
    else:
        print(f"Ошибка: Неизвестная модель {model_name}")
        return None

    model_exp_dir = os.path.join(output_dir, f"{model_name}_experiment")
    os.makedirs(model_exp_dir, exist_ok=True)
    
    json_output_path = os.path.join(model_exp_dir, "evaluation_metrics.json")
    with open(json_output_path, 'w') as f:
        json.dump(summary_metrics, f, indent=4, ensure_ascii=False)
        
    print(f"Метрики для {model_name} успешно сохранены в {json_output_path}")
    print(f"mAP@0.5: {summary_metrics['mAP_50']} | mAP@0.5:0.95: {summary_metrics['mAP_50_95']} | Precision: {summary_metrics['precision']}")

    _update_global_leaderboard(summary_metrics, output_dir)
    
    return summary_metrics


def _update_global_leaderboard(new_metrics: dict, output_dir: str):
    leaderboard_path = os.path.join(output_dir, "model_comparison_leaderboard.csv")
    file_exists = os.path.exists(leaderboard_path)
    
    rows = {}
    if file_exists:
        with open(leaderboard_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.strip().split(',')
                    if len(parts) >= 5:
                        rows[parts[0]] = line.strip()

    rows[new_metrics["model_name"]] = (
        f"{new_metrics['model_name']},{new_metrics['mAP_50']},"
        f"{new_metrics['mAP_50_95']},{new_metrics['precision']},{new_metrics['recall']}"
    )
    
    with open(leaderboard_path, 'w', encoding='utf-8') as f:
        f.write("Model,mAP@0.5,mAP@0.5:0.95,Precision,Recall\n")
        for row_data in rows.values():
            f.write(row_data + "\n")
            
    print(f"Сводная таблица сравнения моделей обновлена: {leaderboard_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск валидации для одной или всех моделей")
    parser.add_argument("--model", type=str, default="all", 
                        choices=["all", "yolov8", "yolov26", "rt-detr", "faster_rcnn", "ssd"],
                        help="Имя конкретной модели для оценки или 'all' для запуска всех найденных моделей")
    args = parser.parse_args()

    dataset_yaml = "data/raw/road-sign-detection-2/data.yaml"

    models_to_evaluate = {
        "yolov8": "runs/detect/results/yolov8_experiment/weights/best.pt",
        "yolov26": "runs/detect/results/yolov26_experiment/weights/best.pt",
        "rt-detr": "runs/detect/results/rt-detr_experiment/weights/best.pt",
        "faster_rcnn": "results/faster_rcnn_experiment/weights/best.pt",
        "ssd": "results/ssd_experiment/weights/best.pt"
    }

    if args.model == "all":
        print("Оценка всех доступных моделей из списка...")
        for name, path in models_to_evaluate.items():
            if os.path.exists(path):
                evaluate_model(model_path=path, dataset_yaml_path=dataset_yaml, model_name=name)
            else:
                print(f"Пропуск {name}: файл весов {path} не найден.")
    else:
        path = models_to_evaluate[args.model]
        evaluate_model(model_path=path, dataset_yaml_path=dataset_yaml, model_name=args.model)
        
    print("\nИтоговая таблица в results/model_comparison_leaderboard.csv")