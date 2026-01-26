import os

import cv2
import numpy as np


def create_image_with_aruco_markers(width, height, marker_size=250, margin=10):
    # Создаем пустое белое изображение
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Инициализация словаря ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # ID маркеров (можно изменить)
    marker_ids = [0, 1, 2, 3]

    # Рассчитываем позиции маркеров:
    # 1) Верхний маркер (по центру верхней грани)
    top_x = (width - marker_size) // 2
    top_y = margin

    # 2) Нижний маркер (по центру нижней грани)
    bottom_x = (width - marker_size) // 2
    bottom_y = height - margin - marker_size

    # 3) Левый маркер (по центру левой грани)
    left_x = margin
    left_y = (height - marker_size) // 2

    # 4) Правый маркер (по центру правой грани)
    right_x = width - margin - marker_size
    right_y = (height - marker_size) // 2

    positions = [
        (top_x, top_y),  # Верхний
        (bottom_x, bottom_y),  # Нижний
        (left_x, left_y),  # Левый
        (right_x, right_y)  # Правый
    ]

    # Генерируем и размещаем маркеры
    for marker_id, (x, y) in zip(marker_ids, positions):
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        marker_image = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)
        image[y:y + marker_size, x:x + marker_size] = marker_image

    return image


# Параметры изображения
width, height = 1024, 1024  # Размер изображения
base_name = 'outs/aruco_markers_test'
extension = '.png'
counter = 1
output_filename = base_name + extension

while os.path.exists(output_filename):
    output_filename = f"{base_name}_{counter}{extension}"
    counter += 1

# Создаем изображение
result_image = create_image_with_aruco_markers(width, height)

# Сохраняем результат
cv2.imwrite(output_filename, result_image)
print(f"Изображение с маркерами сохранено в {output_filename}")

# Показываем результат
cv2.imshow("ArUco Markers (Centered)", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()