import cv2
import time
import os
from datetime import datetime

def capture_photos_every_5_seconds():
    # Создаем папку для сохранения изображений
    output_dir = "saved_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана папка: {output_dir}")
    
    # Настройки интервала
    capture_interval = 10  # Интервал между снимками в секундах
    
    # Инициализация видеозахвата
    cap = cv2.VideoCapture(0)
    
    # Проверка подключения камеры
    if not cap.isOpened():
        print("Ошибка: Не удалось подключиться к камере!")
        print("Попробуйте изменить индекс камеры в cv2.VideoCapture(0) на 1, 2 и т.д.")
        return
    
    # Установка параметров камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Получение фактических параметров
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("=" * 50)
    print("АВТОМАТИЧЕСКАЯ СЪЕМКА КАЖДЫЕ 5 СЕКУНД")
    print("=" * 50)
    print(f"Разрешение: {actual_width}x{actual_height}")
    print(f"FPS: {actual_fps:.1f}")
    print(f"Интервал съемки: каждые {capture_interval} секунд")
    print(f"Папка для сохранения: {output_dir}/")
    print("=" * 50)
    print("Управление:")
    print("  'q' - остановить съемку")
    print("=" * 50)
    
    # Переменные для управления
    capturing = True
    image_count = 0
    last_capture_time = 0
    start_time = time.time()
    
    # Получаем список уже существующих файлов для продолжения нумерации
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
    if existing_files:
        existing_files.sort()
        try:
            # Ищем максимальный номер в существующих файлах
            numbers = []
            for f in existing_files:
                if f.startswith('photo_') and f.endswith('.jpg'):
                    try:
                        num = int(f.split('_')[1].split('.')[0])
                        numbers.append(num)
                    except:
                        pass
            if numbers:
                image_count = max(numbers) + 1
                print(f"Найдено существующих изображений: {len(existing_files)}")
                print(f"Продолжаем с номера: {image_count}")
        except:
            print("Не удалось определить последний номер файла, начинаем с 0")
    
    print(f"\nСъемка начата в {datetime.now().strftime('%H:%M:%S')}")
    print("Ожидание первого снимка...")
    
    # Переменная для отслеживания последнего вывода в консоль
    last_console_output = time.time()
    
    while capturing:
        ret, frame = cap.read()
        
        if not ret:
            print("Ошибка: Не удалось получить кадр с камеры!")
            break
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Выводим отсчет каждую секунду
        if current_time - last_console_output >= 1.0:
            time_until_next = capture_interval - (current_time - last_capture_time)
            if time_until_next > 0:
                print(f"До следующего снимка: {time_until_next:.1f} секунд")
            last_console_output = current_time
        
        # Проверяем, пора ли делать снимок
        if current_time - last_capture_time >= capture_interval:
            # Генерируем имя файла с timestamp и порядковым номером
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"photo_{image_count:04d}_{timestamp}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Сохраняем изображение
            cv2.imwrite(filepath, frame)
            
            # Выводим отчет в консоль
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Снимок #{image_count:04d} сохранен: {filename} "
                  f"(Время: {elapsed_time:.1f}s)")
            
            image_count += 1
            last_capture_time = current_time
            last_console_output = current_time  # Сбрасываем таймер вывода после снимка
        
        # Отображение информации на кадре
        time_until_next = max(0, capture_interval - (current_time - last_capture_time))
        info_text = [
            f"Photos: {image_count}",
            f"Time: {elapsed_time:.1f}s",
            f"Next in: {time_until_next:.1f}s",
            "Press 'q' to quit"
        ]
        
        # Создаем копию кадра для отображения текста
        display_frame = frame.copy()
        for i, text in enumerate(info_text):
            cv2.putText(display_frame, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Отображение видео
        cv2.imshow('Auto Photo Capture - Every 5 Seconds', display_frame)
        
        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            capturing = False
    
    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()
    
    # Финальный отчет
    print("\n" + "=" * 50)
    print("СЪЕМКА ЗАВЕРШЕНА!")
    print("=" * 50)
    print(f"Общее время работы: {elapsed_time:.1f} секунд")
    print(f"Всего сделано снимков: {image_count}")
    print(f"Изображения сохранены в папку: {output_dir}/")
    print("=" * 50)

if __name__ == "__main__":
    capture_photos_every_5_seconds()
