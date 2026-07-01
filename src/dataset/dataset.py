import os
import yaml

class RoadSignDataset:
    def __init__(self, data_yaml_path: str):
        self.config_path = data_yaml_path
        self.dataset_info = self._load_yaml()
        
    def _load_yaml(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Файл конфигурации данных не найден: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def verify_dataset_structure(self):
        base_path = os.path.dirname(self.config_path)
        
        splits = {
            'train': self.dataset_info.get('train', 'train/images'),
            'val': self.dataset_info.get('val', 'valid/images')
        }
        
        for split_name, split_path in splits.items():
            full_images_path = os.path.normpath(os.path.join(base_path, split_path))
            full_labels_path = full_images_path.replace('images', 'labels')
            
            if not os.path.exists(full_images_path):
                print(f"Ошибка: Директория изображений {split_name} не найдена: {full_images_path}")
                return False
                
            if not os.path.exists(full_labels_path):
                print(f"Ошибка: Директория аннотаций {split_name} не найдена: {full_labels_path}")
                return False
                
            num_imgs = len([f for f in os.listdir(full_images_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            num_lbls = len([f for f in os.listdir(full_labels_path) if f.endswith('.txt')])
            
            print(f"Набор [{split_name}]: Изображений = {num_imgs}, Аннотаций (BBox) = {num_lbls}")
            
            if num_imgs != num_lbls:
                print(f"Предупреждение: Количество картинок и файлов разметки в '{split_name}' не совпадает")
                
        print(f"Всего классов для детектирования и классификации: {self.dataset_info.get('nc')}")
        print(f"Имена классов: {list(self.dataset_info.get('names').values())}")
        return True

def prepare_data(data_yaml_path: str):
    return
    dataset = RoadSignDataset(data_yaml_path)
    is_valid = dataset.verify_dataset_structure()
    if not is_valid:
        raise ValueError("Структура датасета нарушена. Исправьте пути в configs/default.yaml")
    return dataset.dataset_info