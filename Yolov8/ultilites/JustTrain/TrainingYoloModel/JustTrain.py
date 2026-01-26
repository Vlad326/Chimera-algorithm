from ultralytics import YOLO
import os
import torch

def train_yolov8():
    # Проверяем доступность GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Загрузка модели (можно выбрать разные размеры: n, s, m, l, x)
    model = YOLO('yolov8n.pt')  # или yolov8s.pt, yolov8m.pt и т.д.

    # Параметры обучения
    train_params = {
        'data': 'yolo_dataset/data.yaml',
        'epochs': 50,
        'batch': 8,
        'imgsz': [640, 640],   #[720, 1280]
        'device': device,
        'workers': 8,
        'optimizer': 'auto',  # можно выбрать SGD, Adam, AdamW, NAdam и др.
    }

    # Запуск обучения
    results = model.train(**train_params)

    # Сохранение лучшей модели
    best_model_path = os.path.join(model.trainer.save_dir, 'weights/best.pt')
    print(f"Best model saved to: {best_model_path}")

    return best_model_path

if __name__ == "__main__":
    trained_model = train_yolov8()
