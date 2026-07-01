import os
import json
from ultralytics import YOLO

def evaluate_model(model_path: str, dataset_yaml_path: str, model_name: str, output_dir: str = "results"):
    print(f"Оценка качества модели {model_name}")
    
    if not os.path.exists(model_path):
        print(f"Ошибка: Веса модели не найдены по пути {model_path}")
        return None
        
    model = YOLO(model_path)
    
    metrics = model.val(
        data=dataset_yaml_path,
        imgsz=416,
        split='val',
        plots=False
    )
    
    summary_metrics = {
        "model_name": model_name,
        "precision": round(metrics.results_dict.get("metrics/precision(B)", 0.0), 4),
        "recall": round(metrics.results_dict.get("metrics/recall(B)", 0.0), 4),
        "mAP_50": round(metrics.results_dict.get("metrics/mAP50(B)", 0.0), 4),
        "mAP_50_95": round(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0), 4),
        "fitness": round(metrics.fitness, 4)
    }
    
    model_exp_dir = os.path.join(output_dir, f"{model_name}_experiment")
    os.makedirs(model_exp_dir, exist_ok=True)
    
    json_output_path = os.path.join(model_exp_dir, "evaluation_metrics.json")
    with open(json_output_path, 'w') as f:
        json.dump(summary_metrics, f, indent=4, ensure_ascii=False)
        
    print(f"Метрики для {model_name} успешно сохранены в {json_output_path}")
    print(f"mAP@0.5: {summary_metrics['mAP_50']} | Precision: {summary_metrics['precision']}")

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
                    if len(parts) == 5:
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