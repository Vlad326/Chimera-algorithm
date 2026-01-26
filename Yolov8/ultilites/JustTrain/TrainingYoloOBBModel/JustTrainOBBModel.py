from ultralytics import YOLO
import yaml
from pathlib import Path
import time


# 1. Подготовка конфигурационного файла
# 2. Обучение модели
def train_yolov8_obb(data_yaml, epochs=100, imgsz=640, batch=8):
    # Загрузка предобученной модели (например, yolov8s-obb)
    model = YOLO('yolov8n-obb.pt')

    # Параметры обучения
    train_args = {
        'data': str(data_yaml),
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'save': True,
        'save_period': 10,
        'workers': 4,
        'optimizer': 'auto',
        'lr0': 0.01,  # Начальная скорость обучения
        'name': 'yolov8n_robot2_obb',
        'patience': 15  # ← ДОБАВЛЕНО ЗДЕСЬ! Ранняя остановка
    }

    # Запуск обучения
    results = model.train(**train_args)

    return results


# 3. Валидация модели
def validate_model(model_path, data_yaml):
    model = YOLO(model_path)
    metrics = model.val(data=str(data_yaml))
    return metrics


if __name__ == "__main__":
    s = time.time()
    
    # Подготовка data.yaml
    data_yaml = "C:/Users/user/Desktop/alghoritmXimeraV1/Yolov8/ultilites/JustTrain/TrainingYoloOBBModel/datasets/Ximera-17-09/data.yaml"

    # Параметры обучения
    EPOCHS = 125 # 75
    IMG_SIZE = [720, 1280]     #[1080, 1920]
    BATCH_SIZE = 6

    # Обучение модели
    print("Starting training...")
    train_results = train_yolov8_obb(data_yaml, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH_SIZE)

    # Путь к лучшей модели
    best_model = Path("runs/obb/yolov8s_robot2_obb/weights/best.pt")

    # Валидация
    if best_model.exists():
        print("\nValidating best model...")
        val_metrics = validate_model(best_model, data_yaml)
        print(f"Validation results: {val_metrics}")
    else:
        print("Training failed - no best model found")

    e = time.time()

    print("---")
    print(f"Training time: {e - s:.2f} seconds")
    print(f"Start: {s}, End: {e}")
