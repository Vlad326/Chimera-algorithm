from ultralytics import YOLO
import yaml
from pathlib import Path
import time


# 1. Подготовка конфигурационного файла
# 2. Обучение модели
def train_yolov8_obb(data_yaml, epochs=100, imgsz=640, batch=8):
    # Загрузка предобученной модели
    model = YOLO('yolov8n-obb.pt')

    # Параметры обучения с исправленной аугментацией
    train_args = {
        'data': str(data_yaml),
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'save': True,
        'save_period': 10,
        'workers': 4,
        'optimizer': 'auto',  # Автоматический подбор оптимизатора
        'name': 'yolov8n_robot2_obb',
        'patience': 50,  # Ранняя остановка
        
        # ИСПРАВЛЕННАЯ АУГМЕНТАЦИЯ
        'augment': True,  # Включение аугментации
        'degrees': 10.0,  # Поворот: ±10 градусов
        'translate': 0.1,  # Сдвиг: 10% от размера изображения
        'scale': 0.2,  # Масштабирование: ±20%
        'shear': 2.0,  # Наклон: ±2 градуса
        'perspective': 0.0005,  # Перспективные искажения
        'fliplr': 0.5,  # Горизонтальное отражение с вероятностью 50%
        'flipud': 0.1,  # Вертикальное отражение с вероятностью 10%
        'mosaic': 0.7,  # Мозаика с вероятностью 70%
        'mixup': 0.1,  # MixUp аугментация с вероятностью 10%
        'copy_paste': 0.1,  # Копирование объектов с вероятностью 10%
        'hsv_h': 0.015,  # Изменение оттенка
        'hsv_s': 0.7,  # Изменение насыщенности
        'hsv_v': 0.4,  # Изменение яркости
        'erasing': 0.4,  # Стирание частей изображения
        
        # УДАЛЕН deprecated параметр 'crop_fraction'
        # ДОБАВЛЕНЫ параметры для лучшей стабильности
        'close_mosaic': 10,  # Отключение mosaic за 10 эпох до конца
        'warmup_epochs': 3.0,  # Прогрев обучения
        'warmup_momentum': 0.8,  # Моментум во время прогрева
        'warmup_bias_lr': 0.1,  # Скорость обучения для bias во время прогрева
        
        # ФИКСИРОВАННЫЙ РАЗМЕР ИЗОБРАЖЕНИЯ
        'rect': False,  # Отключаем rectangular training для точного размера
    }

    # Запуск обучения
    results = model.train(**train_args)

    return results


# 3. Валидация модели
def validate_model(model_path, data_yaml):
    model = YOLO(model_path)
    metrics = model.val(data=str(data_yaml))
    return metrics


# Функция для мониторинга прогресса обучения
def print_training_info(img_size):
    print("=" * 50)
    print("ИНФОРМАЦИЯ О ОБУЧЕНИИ:")
    print("- Модель: YOLOv8n-OBB")
    print(f"- Размер изображения: {img_size}")
    print("=" * 50)


if __name__ == "__main__":
    s = time.time()
    
    # Подготовка data.yaml
    data_yaml = "C:/Users/User\PycharmProjects\PythonProject\Project/algoritmXimeraV1\Yolov8/ultilites\JustTrain\TrainingYoloOBBModel\datasets\Wolfram\data.yaml"

    # Параметры обучения - ФИКСИРОВАННЫЙ РАЗМЕР 1280x720
    EPOCHS = 300
    IMG_SIZE = [960, 1280]  # [height, width] - 720p формат
    BATCH_SIZE = 4

    # Вывод информации о обучении
    print_training_info(f"{IMG_SIZE[1]}x{IMG_SIZE[0]}")  # 1280x720
    
    # Проверка доступности GPU
    import torch
    print(f"CUDA доступен: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Обучение модели
    print("Starting training...")
    train_results = train_yolov8_obb(data_yaml, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH_SIZE)

    # Путь к лучшей модели
    best_model = Path("runs/obb/yolov8n_robot2_obb/weights/best.pt")

    # Валидация
    if best_model.exists():
        print("\nValidating best model...")
        val_metrics = validate_model(best_model, data_yaml)
        print(f"Validation results: {val_metrics}")
        
        # Сохранение метрик в файл
        results_file = Path("training_results.txt")
        with open(results_file, 'w') as f:
            f.write(f"Training completed successfully!\n")
            f.write(f"Image size: {IMG_SIZE[1]}x{IMG_SIZE[0]}\n")
            f.write(f"Best model: {best_model}\n")
            f.write(f"Validation metrics: {val_metrics}\n")
            f.write(f"Total training time: {time.time() - s:.2f} seconds\n")
    else:
        print("Training failed - no best model found")
        # Поиск альтернативных путей
        possible_paths = list(Path("runs/obb").glob("**/best.pt"))
        if possible_paths:
            print(f"Found possible model at: {possible_paths[0]}")
            best_model = possible_paths[0]

    e = time.time()

    print("---")
    print(f"Training time: {(e - s) / 60:.2f} minutes")
    print(f"Start: {time.ctime(s)}, End: {time.ctime(e)}")
