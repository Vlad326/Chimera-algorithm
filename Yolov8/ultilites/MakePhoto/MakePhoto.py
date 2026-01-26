import cv2
import time
import os
import json
import numpy as np
from datetime import datetime
import glob

def load_camera_calibration(calib_file_path):
    """
    Загружает параметры калибровки камеры из JSON файла
    """
    if not os.path.exists(calib_file_path):
        print(f"Ошибка: Файл калибровки {calib_file_path} не найден!")
        print("Запустите сначала скрипт калибровки камеры")
        return None, None, None
    
    try:
        with open(calib_file_path, 'r') as f:
            calib_data = json.load(f)
        
        K = np.array(calib_data['K'])  # Матрица камеры
        D = np.array(calib_data['D'])  # Коэффициенты дисторсии
        imSize = tuple(calib_data['imSize'])  # Размер изображения
        
        print("=" * 50)
        print("ПАРАМЕТРЫ КАЛИБРОВКИ ЗАГРУЖЕНЫ")
        print("=" * 50)
        print(f"Матрица камеры (K):")
        print(K)
        print(f"\nКоэффициенты дисторсии (D):")
        print(D)
        print(f"\nРазмер изображения: {imSize}")
        print("=" * 50)
             
        return K, D, imSize
        
    except Exception as e:
        print(f"Ошибка при загрузке файла калибровки: {e}")
        return None, None, None

def undistort_image(image, K, D):
    """
    Применяет коррекцию дисторсии к изображению
    """
    h, w = image.shape[:2]
    
    # Получаем новые параметры камеры после коррекции
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    
    # Исправляем дисторсию
    undistorted = cv2.undistort(image, K, D, None, new_K)
    
    # Обрезаем изображение по области интереса
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted

def extract_frames_from_video(video_path, output_dir, interval_seconds=5, K=None, D=None):
    """
    Извлекает кадры из видео с заданным интервалом
    """
    if not os.path.exists(video_path):
        print(f"Ошибка: Видеофайл {video_path} не найден!")
        return 0
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл!")
        return 0
    
    # Получаем параметры видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Извлечение кадров из видео: {os.path.basename(video_path)}")
    print(f"Длительность: {duration:.1f} секунд, FPS: {fps:.1f}")
    print(f"Интервал извлечения: каждые {interval_seconds} секунд")
    
    # Рассчитываем интервал в кадрах
    frame_interval = int(fps * interval_seconds)
    if frame_interval == 0:
        frame_interval = 1
    
    image_count = 0
    frame_number = 0
    
    # Получаем список уже существующих файлов для продолжения нумерации
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
    if existing_files:
        existing_files.sort()
        try:
            last_number = int(existing_files[-1].split('_')[1].split('.')[0])
            image_count = last_number + 1
            print(f"Найдено существующих изображений: {len(existing_files)}")
            print(f"Продолжаем с номера: {image_count}")
        except:
            print("Не удалось определить последний номер файла, начинаем с 0")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Извлекаем кадр каждые interval_seconds секунд
        if frame_number % frame_interval == 0:
            # Применяем коррекцию дисторсии если параметры доступны
            if K is not None and D is not None:
                processed_frame = undistort_image(frame, K, D)
                undistort_tag = "_undist"
            else:
                processed_frame = frame
                undistort_tag = "_raw"
            
            # Генерируем имя файла
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"img_{image_count:04d}_{timestamp}{undistort_tag}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Сохраняем изображение
            cv2.imwrite(filepath, processed_frame)
            
            # Выводим отчет в консоль
            current_time = frame_number / fps
            status = "СКОРРЕКТИРОВАН" if K is not None and D is not None else "НЕСКОРРЕКТИРОВАН"
            print(f"[{current_time:.1f}s] Снимок #{image_count:04d} {status}: {filename}")
            
            image_count += 1
        
        frame_number += 1
    
    cap.release()
    return image_count

def record_and_extract_frames():
    # Загрузка параметров калибровки
    calib_file = "NOTC:/Users/user/Desktop/alghoritmXimeraV1/CalibrationCamera/CalibParamsOuts/CalibParams_11.json"
    K, D, imSize = load_camera_calibration(calib_file)
    
    if K is None or D is None:
        print("Продолжаем без коррекции дисторсии...")
        use_undistortion = False
    else:
        use_undistortion = True
    
    # Создаем папки для сохранения
    videos_dir = "recorded_videos"
    output_dir = "train_images"
    
    for directory in [videos_dir, output_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Создана папка: {directory}")
    
    # Настройки камеры
    frame_width = 1280
    frame_height = 720
    fps = 30
    capture_interval = 5  # Интервал между извлечением кадров в секундах
    
    # Инициализация видеозахвата
    cap = cv2.VideoCapture(0)

    # Проверка подключения камеры
    if not cap.isOpened():
        print("Ошибка: Не удалось подключиться к камере!")
        print("Попробуйте изменить индекс камеры в cv2.VideoCapture(0) на 1, 2 и т.д.")
        return
    
    # Установка параметров камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Настройки для лучшего качества изображения
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Ручной режим экспозиции
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # Уменьшение экспозиции (меньше размытие)
    cap.set(cv2.CAP_PROP_GAIN, 75)             # Увеличение усиления для компенсации
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)       # Средняя яркость
    cap.set(cv2.CAP_PROP_CONTRAST, 70)         # Увеличенная контрастность
    cap.set(cv2.CAP_PROP_SATURATION, 70)       # Увеличенная насыщенность
    cap.set(cv2.CAP_PROP_SHARPNESS, 100)       # Максимальная резкость
    
    # Получение фактических параметров
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Создаем имя для видеофайла
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_filename = f"training_video_{timestamp}.avi"
    video_path = os.path.join(videos_dir, video_filename)
    
    # Определяем кодек и создаем VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, actual_fps, (actual_width, actual_height))
    
    print("=" * 50)
    print("ЗАПИСЬ ВИДЕО ДЛЯ ОБУЧЕНИЯ YOLO")
    print("=" * 50)
    print(f"Коррекция дисторсии: {'ВКЛ' if use_undistortion else 'ВЫКЛ'}")
    print(f"Разрешение: {actual_width}x{actual_height}")
    print(f"FPS: {actual_fps:.1f}")
    print(f"Видео будет сохранено как: {video_filename}")
    print(f"Кадры будут извлекаться каждые: {capture_interval} секунд")
    print(f"Папка для изображений: {output_dir}/")
    print("=" * 50)
    print("Управление:")
    print("  'q' - остановить запись и извлечь кадры")
    print("  'u' - переключить коррекцию дисторсии")
    print("=" * 50)
    
    # Переменные для управления
    recording = True
    undistort_enabled = use_undistortion
    start_time = time.time()
    
    print(f"\nЗапись начата в {datetime.now().strftime('%H:%M:%S')}")
    print("Записываем видео...")
    
    while recording:
        ret, frame = cap.read()
        
        if not ret:
            print("Ошибка: Не удалось получить кадр с камеры!")
            break
        
        # Применяем коррекцию дисторсии для отображения
        if undistort_enabled and K is not None and D is not None:
            processed_frame = undistort_image(frame, K, D)
            processed_frame = cv2.resize(processed_frame, (actual_width, actual_height))
        else:
            processed_frame = frame.copy()
        
        # Записываем оригинальный кадр в видео (без коррекции)
        out.write(frame)
        
        # Отображение информации на кадре
        elapsed_time = time.time() - start_time
        info_text = [
            f"Recording: {elapsed_time:.1f}s",
            f"Undistort: {'ON' if undistort_enabled else 'OFF'}",
            "Press 'q' to stop and extract frames"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(processed_frame, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Отображение видео
        cv2.imshow('Recording Video for YOLO Training', processed_frame)
        
        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            recording = False
        elif key == ord('u'):  # Переключение коррекции дисторсии
            undistort_enabled = not undistort_enabled
            status = "ВКЛ" if undistort_enabled else "ВЫКЛ"
            print(f"Коррекция дисторсии (для отображения): {status}")
    
    # Освобождение ресурсов
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Финальный отчет о записи
    recording_duration = time.time() - start_time
    print("\n" + "=" * 50)
    print("ЗАПИСЬ ЗАВЕРШЕНА!")
    print("=" * 50)
    print(f"Длительность записи: {recording_duration:.1f} секунд")
    print(f"Видео сохранено как: {video_path}")
    
    # Извлекаем кадры из видео
    print("\n" + "=" * 50)
    print("ИЗВЛЕЧЕНИЕ КАДРОВ ИЗ ВИДЕО")
    print("=" * 50)
    
    # Используем параметры калибровки для коррекции при извлечении
    extract_K = K if undistort_enabled else None
    extract_D = D if undistort_enabled else None
    
    extracted_count = extract_frames_from_video(video_path, output_dir, capture_interval, extract_K, extract_D)
    
    print("\n" + "=" * 50)
    print("ПРОЦЕСС ЗАВЕРШЕН!")
    print("=" * 50)
    print(f"Записано видео: {recording_duration:.1f} секунд")
    print(f"Извлечено кадров: {extracted_count}")
    print(f"Видеофайл: {video_path}")
    print(f"Изображения сохранены в: {output_dir}/")
    print("=" * 50)

if __name__ == "__main__":
    record_and_extract_frames()
