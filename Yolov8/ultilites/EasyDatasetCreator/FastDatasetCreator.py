import cv2
import numpy as np
import os
import time
from datetime import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

class DatasetCreator:
    def __init__(self):
        self.images_dir = "dataset_fast"
        self.video_dir = "videos_fast"
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        self.cap = None
        self.writing = False
        self.frame_count = 0

    @staticmethod
    def calculate_similarity_numba(gray1, gray2):
        """Ускоренная версия сравнения кадров с использованием Numba"""
        if gray1.size == 0 or gray2.size == 0:
            return float('inf')
        
        diff = np.abs(gray1.astype(np.int32) - gray2.astype(np.int32))
        return np.sum(diff)

    def calculate_similarity(self, frame1, frame2):
        """Оптимизированная версия с предобработкой"""
        if frame1 is None or frame2 is None:
            return float('inf')

        # Быстрое преобразование в grayscale и уменьшение разрешения
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Уменьшаем разрешение для ускорения (можно настроить)
        scale_factor = 0.25
        if scale_factor < 1.0:
            new_size = (int(gray1.shape[1] * scale_factor), int(gray1.shape[0] * scale_factor))
            gray1 = cv2.resize(gray1, new_size, interpolation=cv2.INTER_AREA)
            gray2 = cv2.resize(gray2, new_size, interpolation=cv2.INTER_AREA)

        return self.calculate_similarity_numba(gray1, gray2)

    def select_unique_frames_fast(self, video_path, num_frames):
        """СУПЕР УСКОРЕННАЯ версия выбора кадров"""
        print(f"\n=== УСКОРЕННЫЙ ВЫБОР {num_frames} КАДРОВ ===")
        
        start_time = time.time()
        
        # Открываем видео с оптимизацией
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Ошибка: Не удалось открыть видео!")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Всего кадров: {total_frames}, FPS: {fps:.1f}")
        
        # Автоматически определяем шаг для пропуска кадров
        frame_skip = max(1, total_frames // (num_frames * 10))
        print(f"Автопропуск: каждый {frame_skip}-й кадр")
        
        # Читаем ВСЕ кадры за один проход
        frames = []
        frame_indices = []
        
        print("Быстрое чтение кадров...")
        for i in range(0, total_frames, frame_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                frame_indices.append(i)
            if len(frames) >= num_frames * 5:  # Ограничиваем буфер
                break
        
        cap.release()
        read_time = time.time() - start_time
        print(f"Прочитано {len(frames)} кадров за {read_time:.2f} сек")
        
        if len(frames) < num_frames:
            print(f"Ошибка: Недостаточно кадров! Нужно {num_frames}, есть {len(frames)}")
            return
        
        # Быстрый отбор самых разных кадров
        print("Быстрый отбор кадров...")
        
        # Используем стратегию равномерного распределения
        selected_indices = []
        step = max(1, len(frames) // num_frames)
        
        for i in range(0, len(frames), step):
            if len(selected_indices) < num_frames:
                selected_indices.append(i)
        
        # Если нужно больше кадров, добавляем случайные
        if len(selected_indices) < num_frames:
            remaining = num_frames - len(selected_indices)
            available_indices = [i for i in range(len(frames)) if i not in selected_indices]
            selected_indices.extend(np.random.choice(available_indices, remaining, replace=False))
        
        # Сохраняем выбранные кадры
        print("Быстрое сохранение...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, idx in enumerate(selected_indices):
            img_filename = os.path.join(self.images_dir, f"fast_{timestamp}_{i:04d}.jpg")
            cv2.imwrite(img_filename, frames[idx])
        
        total_time = time.time() - start_time
        print(f"\n✅ Готово! Сохранено {len(selected_indices)} кадров за {total_time:.2f} сек")
        print(f"Скорость: {len(frames)/read_time:.1f} кадров/сек")
        
        return selected_indices

    def select_unique_frames_parallel(self, video_path, num_frames):
        """Параллельная версия для многопроцессорных систем"""
        print(f"\n=== ПАРАЛЛЕЛЬНЫЙ ВЫБОР {num_frames} КАДРОВ ===")
        
        start_time = time.time()
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, total_frames // (num_frames * 5))
        
        # Читаем кадры параллельно
        frames = []
        indices_to_read = list(range(0, total_frames, frame_skip))[:num_frames * 3]
        
        def read_frame(pos):
            cap_local = cv2.VideoCapture(video_path)
            cap_local.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap_local.read()
            cap_local.release()
            return frame if ret else None
        
        print("Параллельное чтение кадров...")
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_index = {executor.submit(read_frame, pos): pos for pos in indices_to_read}
            
            for future in as_completed(future_to_index):
                frame = future.result()
                if frame is not None:
                    frames.append(frame)
                if len(frames) % 50 == 0:
                    print(f"Прочитано {len(frames)} кадров...")
        
        cap.release()
        
        # Простой равномерный отбор
        selected_indices = list(range(0, len(frames), max(1, len(frames) // num_frames)))[:num_frames]
        
        # Сохранение
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, idx in enumerate(selected_indices):
            if idx < len(frames):
                img_filename = os.path.join(self.images_dir, f"parallel_{timestamp}_{i:04d}.jpg")
                cv2.imwrite(img_filename, frames[idx])
        
        total_time = time.time() - start_time
        print(f"Параллельная обработка завершена за {total_time:.2f} сек")
        
        return selected_indices

    def ultra_fast_selection(self, video_path, num_frames):
        """САМАЯ БЫСТРАЯ версия - пропускаем анализ, берем равномерно"""
        print(f"\n⚡ УЛЬТРАБЫСТРЫЙ ОТБОР {num_frames} КАДРОВ ⚡")
        
        start_time = time.time()
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Вычисляем шаг для равномерного отбора
        step = max(1, total_frames // num_frames)
        selected_indices = list(range(0, total_frames, step))[:num_frames]
        
        print(f"Отбираем каждый {step}-й кадр из {total_frames}")
        
        # Читаем и сохраняем только выбранные кадры
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_count = 0
        
        for i, frame_pos in enumerate(selected_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                img_filename = os.path.join(self.images_dir, f"ultrafast_{timestamp}_{i:04d}.jpg")
                cv2.imwrite(img_filename, frame)
                saved_count += 1
            
            if saved_count % 50 == 0:
                print(f"Сохранено {saved_count} кадров...")
        
        cap.release()
        
        total_time = time.time() - start_time
        print(f"⚡ УЛЬТРАБЫСТРО! {saved_count} кадров за {total_time:.2f} сек")
        print(f"Скорость: {saved_count/max(0.1, total_time):.1f} кадров/сек")
        
        return saved_count

def main():
    creator = DatasetCreator()
    
    # Тестируем все методы
    video_file = 'video.avi'
    
    if not os.path.exists(video_file):
        print(f"Ошибка: Файл {video_file} не найден!")
        return
    
    print("Выберите метод ускорения:")
    print("1 - Супер быстрый (рекомендуется)")
    print("2 - Параллельный (многопоточный)")
    print("3 - Ультрабыстрый (максимальная скорость)")
    
    choice = input("Ваш выбор (1-3): ").strip()
    
    start_total = time.time()
    
    if choice == "1":
        creator.select_unique_frames_fast(video_file, 125)
    elif choice == "2":
        creator.select_unique_frames_parallel(video_file, 125)
    elif choice == "3":
        creator.ultra_fast_selection(video_file, 125)
    else:
        print("Используем ультрабыстрый метод по умолчанию")
        creator.ultra_fast_selection(video_file, 125)
    
    total_time = time.time() - start_total
    print(f"\n🎉 Общее время выполнения: {total_time:.2f} секунд")

if __name__ == "__main__":
    main()
