import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

SOURCE_DATASET = "yolo_dataset"  # Исходный датасет
TARGET_DATASET = "yolo_dataset_resized_2"  # Выходной датасет
NEW_SIZE = (1280, 720)  # Новый размер (ширина, высота)


def process_image_and_labels(img_path, new_img_path, label_path, new_label_path):
    # Загрузка изображения
    img = cv2.imread(img_path)
    if img is None:
        print(f"Ошибка: Не удалось загрузить {img_path}")
        return

    orig_h, orig_w = img.shape[:2]
    new_w, new_h = NEW_SIZE

    # Растягиваем изображение (без сохранения пропорций)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(new_img_path, img_resized)

    # Если нет файла разметки — создаём пустой
    if not os.path.exists(label_path):
        open(new_label_path, 'w').close()
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        cls, x_center, y_center, width, height = parts
        x_center, y_center, width, height = map(float, [x_center, y_center, width, height])

        # Масштабирование координат bbox
        scale_x = new_w / orig_w  # Масштаб по ширине (например, 640/1920 = 0.333)
        scale_y = new_h / orig_h  # Масштаб по высоте (например, 640/1080 ≈ 0.593)

        # Новые координаты (центр и размеры масштабируются)
        x_center_new = x_center * scale_x
        y_center_new = y_center * scale_y
        width_new = width * scale_x
        height_new = height * scale_y

        # Нормализация к [0, 1] (на случай, если масштабирование вышло за границы)
        x_center_new = np.clip(x_center_new, 0, 1)
        y_center_new = np.clip(y_center_new, 0, 1)
        width_new = np.clip(width_new, 0, 1 - x_center_new)
        height_new = np.clip(height_new, 0, 1 - y_center_new)

        # Пропускаем слишком маленькие объекты
        if width_new < 0.005 or height_new < 0.005:
            continue

        new_line = f"{cls} {x_center_new:.6f} {y_center_new:.6f} {width_new:.6f} {height_new:.6f}\n"
        new_lines.append(new_line)

    # Сохраняем новые аннотации
    with open(new_label_path, 'w') as f:
        f.writelines(new_lines)


# Создаём структуру папок
os.makedirs(os.path.join(TARGET_DATASET, "images"), exist_ok=True)
os.makedirs(os.path.join(TARGET_DATASET, "labels"), exist_ok=True)

# Копируем вспомогательные файлы (data.yaml, labels.cache)
for file in ["data.yaml", "labels.cache"]:
    src = os.path.join(SOURCE_DATASET, file)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(TARGET_DATASET, file))

# Обрабатываем все изображения
img_files = [f for f in os.listdir(os.path.join(SOURCE_DATASET, "images"))
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_file in tqdm(img_files, desc="Обработка изображений"):
    base_name = os.path.splitext(img_file)[0]
    src_img = os.path.join(SOURCE_DATASET, "images", img_file)
    src_label = os.path.join(SOURCE_DATASET, "labels", f"{base_name}.txt")
    dst_img = os.path.join(TARGET_DATASET, "images", img_file)
    dst_label = os.path.join(TARGET_DATASET, "labels", f"{base_name}.txt")
    process_image_and_labels(src_img, dst_img, src_label, dst_label)

print("✅ Готово! Изображения растянуты, bbox отмасштабированы корректно.")
