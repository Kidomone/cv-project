import os
import yaml
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class RoadSignDataset(Dataset):
    def __init__(self, data_yaml_path: str, split: str = 'train'):
        self.config_path = os.path.abspath(data_yaml_path)
        self.dataset_info = self._load_yaml()
        
        base_path = os.path.dirname(self.config_path)
        
        if split == 'train':
            possible_keys = ['train', 'train_images', 'images_train']
        else:
            possible_keys = ['val', 'valid', 'val_images', 'images_val']
            
        split_dir = None
        for key in possible_keys:
            if key in self.dataset_info:
                split_dir = self.dataset_info[key]
                break
                
        if not split_dir:
            split_dir = f'{split}/images' if split == 'train' else 'valid/images'
            
        clean_split_dir = split_dir.replace('../', '').replace('./', '')
        self.images_dir = os.path.abspath(os.path.join(base_path, clean_split_dir))
        
        if 'images' in self.images_dir:
            self.labels_dir = self.images_dir.replace('images', 'labels')
        else:
            self.labels_dir = os.path.join(os.path.dirname(self.images_dir), 'labels')
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(
                f"Директория изображений не найдена!\n"
                f"Искал по абсолютному пути: {self.images_dir}\n"
                f"Файл конфига данных лежит тут: {self.config_path}\n"
                f"Значение ключа из yaml: {split_dir}"
            )
            
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        self.transform = T.Compose([T.ToTensor()])
        
    def _load_yaml(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Файл конфигурации данных не найден: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        lbl_path = os.path.join(self.labels_dir, lbl_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:])
                        
                        xmin = (x_c - w / 2) * width
                        ymin = (y_c - h / 2) * height
                        xmax = (x_c + w / 2) * width
                        ymax = (y_c + h / 2) * height
                        
                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(cls + 1)
                            
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }
        
        if self.transform:
            img = self.transform(img)
            
        return img, target

    def verify_dataset_structure(self):
        base_path = self.dataset_info.get('path', os.path.dirname(self.config_path))
        splits = {
            'train': self.dataset_info.get('train'),
            'val': self.dataset_info.get('val')
        }
        for split_name, split_path in splits.items():
            if not split_path:
                continue
            full_images_path = os.path.normpath(os.path.join(base_path, split_path))
            if 'images' in full_images_path:
                full_labels_path = full_images_path.replace('images', 'labels')
            else:
                full_labels_path = os.path.join(os.path.dirname(full_images_path), 'labels')
                
            if not os.path.exists(full_images_path) or not os.path.exists(full_labels_path):
                return False
            num_imgs = len([f for f in os.listdir(full_images_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            num_lbls = len([f for f in os.listdir(full_labels_path) if f.endswith('.txt')])
            print(f"Набор [{split_name}]: Изображений = {num_imgs}, Аннотаций = {num_lbls}")
        return True

def prepare_data(data_yaml_path: str):
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Файл конфигурации данных не найден: {data_yaml_path}")
    with open(data_yaml_path, 'r') as f:
        info = yaml.safe_load(f)
    print(f"Всего классов для детектирования и классификации: {info.get('nc')}")
    
    names_data = info.get('names', [])
    if isinstance(names_data, dict):
        class_names = list(names_data.values())
    else:
        class_names = list(names_data)
        
    print(f"Имена классов: {class_names}")
    return info