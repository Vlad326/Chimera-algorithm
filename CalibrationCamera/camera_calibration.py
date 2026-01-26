import numpy as np
import cv2
import json
import os
import glob

base_name = 'CalibParamsOuts/CalibParams'
extension = '.json'
counter = 1
PATH_TO_WRITE = base_name + extension

while os.path.exists(PATH_TO_WRITE):
    PATH_TO_WRITE = f"{base_name}_{counter}{extension}"
    counter += 1

CHESSBOARD_DIMENSION = (10, 7)  # количество клеток шахматной доски
SQUARE_SIZE = 65    # размер qодного квадратика шахматной доски в миллиметрах

# Получаем все PNG-файлы из папки calib_camera
image_paths = sorted(glob.glob('CalibParams/*.jpg'))

if not image_paths:
    print("Не найдено PNG-файлов в папке calib_camera!")
    exit()

# критерий остановки поиска углов шахматной доски на субпиксельном уроqвне
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# координаты углов шахматной доски на плоскости
chdim = CHESSBOARD_DIMENSION
objectPoints = np.zeros((chdim[0] * chdim[1], 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:chdim[0], 0:chdim[1]].T.reshape(-1, 2)
objectPoints = objectPoints * SQUARE_SIZE   # переводим в миллиметры

# сюда будем сохранять найденные точки - углы шахматной доски
image_points = []
valid_images = []

for path in image_paths:    # пробегаем по каждому пути до изображения
    image_raw = cv2.imread(path)     # читаем с диска цветное изображение
    
    if image_raw is None:
        print(f"Не удалось загрузить изображение: {path}")
        continue


    #image_raw = cv2.resize(image_raw, [2592, 1944])
    gray_image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)  # переводим в градации серого

    # Ищем углы шахматной доски на изображении
    ret, corners = cv2.findChessboardCorners(gray_image, CHESSBOARD_DIMENSION)

    if ret:    # если нашли углы
        # Уточняем их координаты на субпиксельном уровне
        corners = cv2.cornerSubPix(gray_image, corners, (10, 10), (-1, -1), criteria)

        image_points.append(corners)
        valid_images.append(path)

        # выводим результат на экран
        cv2.drawChessboardCorners(image_raw, CHESSBOARD_DIMENSION, corners, ret)
        cv2.imshow(f"{path}", image_raw)
        cv2.waitKey(0)
    else:
        print(f"Не найдены углы шахматной доски на изображении: {os.path.basename(path)}")

cv2.destroyAllWindows()

if not image_points:
    print("Не удалось найти углы шахматной доски ни на одном изображении!")
    exit()

# Определяем размер изображения по первому успешно обработанному изображению
sample_image = cv2.imread(valid_images[0], cv2.IMREAD_GRAYSCALE)
imageSize = (sample_image.shape[1], sample_image.shape[0])
imageNum = len(image_points)  # количество изображений

print(f"\nУспешно обработано {imageNum} изображений из {len(image_paths)}")

# Калибруем одну камеру
ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
    [objectPoints]*imageNum, 
    image_points, 
    imageSize, 
    None, 
    None
)

print('\nМатрица камеры:')
print(K)
print('\nКоэффициенты дисторсии:')
print(D)
print('\nОценка(ошибка) точности найденных параметров:')
print(ret)

calibParam = {
    "K": K.tolist(),
    "D": D.tolist(),
    "imSize": imageSize,
    "used_images": valid_images
}

with open(PATH_TO_WRITE, 'w') as fp:
    json.dump(calibParam, fp, indent=4)     # записываем найденные калибровочные параметры в json-файл

print(f"\nПараметры калибровки сохранены в файл: {PATH_TO_WRITE}")
