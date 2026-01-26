import os
import cv2
import numpy as np


def generate_fullpage_chessboard(rows, cols, square_size=50, invert_colors=False):
    """
    Генерирует шахматную доску на весь лист без отступов.

    Параметры:
        rows (int): Количество внутренних углов по вертикали
        cols (int): Количество внутренних углов по горизонтали
        square_size (int): Размер квадрата в пикселях
        invert_colors (bool): Инвертировать цвета клеток

    Возвращает:
        np.ndarray: Изображение шахматной доски (BGR)
    """
    # Рассчитываем размер изображения (количество клеток = углы + 1)
    width = (cols + 1) * square_size
    height = (rows + 1) * square_size

    # Создаем изображение
    chessboard = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Рисуем клетки
    for i in range(rows + 1):
        for j in range(cols + 1):
            x = j * square_size
            y = i * square_size

            # Определяем цвет клетки
            if (i + j) % 2 == 0:
                color = (255, 255, 255) if invert_colors else (0, 0, 0)
            else:
                color = (0, 0, 0) if invert_colors else (255, 255, 255)

            # Закрашиваем квадрат
            chessboard[y:y + square_size, x:x + square_size] = color

    return chessboard


# Параметры доски
rows, cols = 12, 15  # Количество внутренних углов (стандарт для калибровки)
square_size = 100  # Размер клетки в пикселях

# Генерация доски
chessboard = generate_fullpage_chessboard(rows, cols, square_size)

# Автоматическое имя файла
base_name = 'outs/ChessBoard'
extension = '.png'
counter = 1
output_filename = base_name + extension

while os.path.exists(output_filename):
    output_filename = f"{base_name}_{counter}{extension}"
    counter += 1

# Сохранение
cv2.imwrite(output_filename, chessboard)
print(f"Шахматная доска сохранена в {output_filename}")

# Отображение
cv2.imshow("Chessboard", chessboard)
cv2.waitKey(0)
cv2.destroyAllWindows()