# src/training/train.py
import os
import yaml
from src.models.models import get_model
from src.dataset.dataset import prepare_data
from src.utils.utils import visualize_random_prediction
from src.evaluation.metrics import evaluate_model

def run_training(model_name: str, config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    prepare_data(config['data_config'])
    model = get_model(model_name)
    exp_name = f"{model_name}_experiment"
    
    model.train(
        data=config['data_config'],
        epochs=config['epochs'],
        batch=config['batch_size'],
        imgsz=config['img_size'],
        patience=config['patience'],
        device=config['device'],
        project=config['project_dir'],
        name=exp_name,
        exist_ok=True
    )
    
    best_model_path = os.path.join(config['project_dir'], exp_name, "weights", "best.pt")
    
    visualize_random_prediction(model_path=best_model_path, dataset_yaml_path=config['data_config'])
    
    evaluate_model(
        model_path=best_model_path,
        dataset_yaml_path=config['data_config'],
        model_name=model_name,
        output_dir=config['project_dir']
    )