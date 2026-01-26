import cv2
import numpy as np
import os
import time
from datetime import datetime
import argparse


class DatasetCreator:
    def __init__(self):
        self.images_dir = "dataset"
        self.video_dir = "videos"  # Добавил отсутствующую переменную
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)  # Создаем папку для видео

        self.cap = None
        self.writing = False
        self.frame_count = 0

    def calculate_similarity(self, frame1, frame2):
        """Вычисляет меру различия между двумя кадрами"""
        if frame1 is None or frame2 is None:
            return float('inf')

        # Преобразуем в оттенки серого для упрощения
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Вычисляем абсолютную разницу
        diff = cv2.absdiff(gray1, gray2)

        # Возвращаем сумму различий (чем больше, тем более разные кадры)
        return np.sum(diff)

    def record_video(self):
        """Этап 1: Запись видео"""
        print("=== ЭТАП 1: ЗАПИСЬ ВИДЕО ===")
        print("Нажмите 's' чтобы начать запись")
        print("Нажмите 'q' чтобы закончить запись и перейти к следующему этапу")
        print("Нажмите 'ESC' для выхода")

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Ошибка: Не удалось открыть камеру!")
            return False

        # Настройки видео
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        fps = 30

        # Создаем имя файла с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(self.video_dir, f"recording_{timestamp}.avi")

        # Настройка видеозаписи
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = None

        cv2.namedWindow("Запись видео - Нажмите 's' чтобы начать", cv2.WINDOW_NORMAL)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Ошибка: Не удалось получить кадр с камеры!")
                break

            # Отображаем статус записи
            if self.writing:
                cv2.putText(frame, "ЗАПИСЬ...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frame_count_text = f"Кадров: {self.frame_count}"
                cv2.putText(frame, frame_count_text, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Запись видео - Нажмите 's' чтобы начать", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and not self.writing:
                # Начинаем запись
                self.writing = True
                out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
                print("Запись начата...")

            elif key == ord('q') and self.writing:
                # Заканчиваем запись
                break

            elif key == 27:  # ESC
                print("Запись прервана")
                if out:
                    out.release()
                self.cap.release()
                cv2.destroyAllWindows()
                return False

            # Записываем кадр если идет запись
            if self.writing:
                out.write(frame)
                self.frame_count += 1

        # Завершение записи
        if out:
            out.release()
        self.cap.release()
        cv2.destroyAllWindows()

        print(f"Запись завершена! Сохранено {self.frame_count} кадров в {video_filename}")
        return True

    def select_unique_frames(self, video_path, num_frames):
        """Этап 2: Выбор уникальных кадров из видео"""
        print(f"\n=== ЭТАП 2: ВЫБОР {num_frames} УНИКАЛЬНЫХ КАДРОВ ===")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            print("Ошибка: Видео не содержит кадров!")
            return

        print(f"Всего кадров в видео: {total_frames}")

        # Считываем каждый 6-й кадр для оптимизации
        frames = []
        print("Чтение каждого 6-го кадра из видео...")
        
        frame_skip = 16  # Обрабатываем каждый 6-й кадр
        frames_processed = 0
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if ret and i % frame_skip == 0:  # Берем только каждый 6-й кадр
                frames.append(frame)
                frames_processed += 1
                if frames_processed % 100 == 0:  # Прогресс каждые 100 кадров
                    print(f"Обработано {frames_processed} кадров...")

        cap.release()

        print(f"Всего обработано кадров после пропуска: {len(frames)}")

        if len(frames) < num_frames:
            print(f"Ошибка: После пропуска осталось только {len(frames)} кадров, но запрошено {num_frames}!")
            return

        # Выбираем самые уникальные кадры
        print("Выбор самых уникальных кадров...")

        # Первый кадр всегда включаем (обычно самый статичный)
        selected_indices = [0]
        selected_frames = [frames[0]]

        # Пока не наберем нужное количество кадров
        while len(selected_indices) < num_frames:
            max_similarity = -1
            best_index = -1

            # Ищем кадр, который максимально отличается от всех выбранных
            for i in range(1, len(frames)):
                if i in selected_indices:
                    continue

                # Вычисляем минимальное сходство с уже выбранными кадрами
                min_similarity_to_selected = float('inf')
                for selected_frame in selected_frames:
                    similarity = self.calculate_similarity(frames[i], selected_frame)
                    min_similarity_to_selected = min(min_similarity_to_selected, similarity)

                # Выбираем кадр с максимальным отличием
                if min_similarity_to_selected > max_similarity:
                    max_similarity = min_similarity_to_selected
                    best_index = i

            if best_index != -1:
                selected_indices.append(best_index)
                selected_frames.append(frames[best_index])
                print(f"Выбран кадр {best_index + 1}/{len(frames)} (реальный номер: {best_index * 6 + 1})")

        # Сохраняем выбранные кадры
        print("Сохранение выбранных кадров...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, idx in enumerate(selected_indices):
            img_filename = os.path.join(self.images_dir, f"image_{timestamp}_{i + 1:04d}.jpg")
            cv2.imwrite(img_filename, frames[idx])
            print(f"Сохранен: {img_filename} (оригинальный кадр: {idx * 6})")

        print(f"\nУспешно сохранено {len(selected_indices)} уникальных кадров!")
        print(f"Оптимизация: обработан каждый 6-й кадр, что в 6 раз быстрее!")

        return selected_indices

    def process_video_file(self, video_path):
        """Обработка существующего видеофайла"""
        print(f"Обработка видеофайла: {video_path}")

        try:
            num_frames = int(input("Введите количество уникальных кадров для выборки: "))
            self.select_unique_frames(video_path, num_frames)
        except ValueError:
            print("Ошибка: Введите корректное число!")
        except Exception as e:
            print(f"Ошибка при обработке видео: {e}")

    def run(self):
        """Основной цикл программы"""
        print("=" * 50)
        print("СОЗДАТЕЛЬ ДАТАСЕТА ДЛЯ YOLOv8-OBB")
        print("=" * 50)

        while True:
            print("\nВыберите действие:")
            print("1 - Записать новое видео и выбрать кадры")
            print("2 - Обработать существующее видео")
            print("3 - Выход")

            choice = input("Ваш выбор (1-3): ").strip()

            if choice == "1":
                if self.record_video():
                    # Получаем последний записанный видеофайл
                    video_files = sorted([f for f in os.listdir(self.video_dir) if f.endswith('.avi')])
                    if video_files:
                        latest_video = os.path.join(self.video_dir, video_files[-1])
                        try:
                            num_frames = int(input("Введите количество уникальных кадров для выборки: "))
                            self.select_unique_frames(latest_video, num_frames)
                        except ValueError:
                            print("Ошибка: Введите корректное число!")
                    else:
                        print("Не найдено записанных видеофайлов!")

            elif choice == "2":
                video_files = [f for f in os.listdir(self.video_dir) if f.endswith('.avi')]
                if video_files:
                    print("Доступные видеофайлы:")
                    for i, f in enumerate(video_files, 1):
                        print(f"{i} - {f}")

                    try:
                        vid_choice = int(input("Выберите видеофайл (номер): ")) - 1
                        if 0 <= vid_choice < len(video_files):
                            self.process_video_file(os.path.join(self.video_dir, video_files[vid_choice]))
                        else:
                            print("Неверный выбор!")
                    except ValueError:
                        print("Ошибка: Введите корректный номер!")
                else:
                    print("В папке videos нет видеофайлов!")

            elif choice == "3":
                print("Выход из программы...")
                break

            else:
                print("Неверный выбор! Попробуйте снова.")


def main():

    creator = DatasetCreator()

    # Если переданы аргументы командной строки
    if True:
        creator.select_unique_frames('video.avi', 125)
    else:
        # Интерактивный режим
        creator.run()


if __name__ == "__main__":
    main()
