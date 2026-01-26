import cv2
import os

# Создаем папку для сохранения кадров
output_folder = "train_images"
os.makedirs(output_folder, exist_ok=True)

# Загружаем видео
video_path = "out_2.avi"  # Укажите путь к вашему видео
cap = cv2.VideoCapture(video_path)

# Проверяем, открылось ли видео
if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)  # Количество кадров в секунду
frame_interval = 1
frame_count = 0
saved_count = 126

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Видео закончилось
    
    # Сохраняем кадр каждые 150 мс
    current_time = frame_count / fps
    if frame_count % int(fps * frame_interval) == 0:
        frame_name = f"{output_folder}/frame_{saved_count:04d}.jpg"
        cv2.imwrite(frame_name, frame)
        saved_count += 1
        print(f"Сохранен кадр: {frame_name}")
    
    frame_count += 1

cap.release()
print(f"Готово! Сохранено {saved_count} кадров в {output_folder}.")
