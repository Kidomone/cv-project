import os
import yaml
import torch
from torch.utils.data import DataLoader
from src.models.models import get_model
from src.dataset.dataset import prepare_data
from src.utils.utils import visualize_random_prediction
from src.evaluation.metrics import evaluate_model

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    dataset_len = len(data_loader)
    epoch_loss = 0
    
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        epoch_loss += losses.item()
        
    return epoch_loss / dataset_len

def run_training(model_name: str, config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    prepare_data(config['data_config'])
    num_classes = 4 
    model = get_model(model_name, num_classes=num_classes)
    exp_name = f"{model_name}_experiment"
    
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)
    
    weights_dir = os.path.join(config['project_dir'], exp_name, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    best_model_path = os.path.join(weights_dir, "best.pt")

    if model_name.lower() in ['yolov8', 'yolo26', 'rt-detr']:
        model.train(
            data=config['data_config'],
            epochs=config['epochs'],
            batch=4,
            imgsz=config['img_size'],
            patience=config['patience'],
            device=config['device'],
            project=config['project_dir'],
            name=exp_name,
            exist_ok=True
        )
    else:
        try:
            from src.dataset.dataset import RoadSignDataset
            
            with open(config['data_config'], 'r') as df:
                data_yaml = yaml.safe_load(df)
            
            train_dataset = RoadSignDataset(config['data_config'], split='train')
            train_loader = DataLoader(
                train_dataset, 
                batch_size=16, 
                shuffle=True, 
                num_workers=2, 
                collate_fn=collate_fn
            )
        except Exception as e:
            raise RuntimeError(
                f"Ошибка инициализации DataLoader для PyTorch. "
                f"Проверь структуру src/dataset/dataset.py: {e}"
            )

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
        
        epochs = config['epochs']
        best_loss = float('inf')
        
        for epoch in range(epochs):
            loss = train_one_epoch(model, optimizer, train_loader, device)
            print(f"Epoch [{epoch+1}/{epochs}] — Loss: {loss:.4f}")

            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Сохранены новые лучшие веса в {best_model_path}")
                
        print(f"Обучение {model_name} завершено")
        
    visualize_random_prediction(
    model_path=best_model_path, 
    dataset_yaml_path=config['data_config'],
    model_name=model_name
    )
    
    evaluate_model(
        model_path=best_model_path,
        dataset_yaml_path=config['data_config'],
        model_name=model_name,
        output_dir=config['project_dir']
    )