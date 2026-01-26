import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class DistortionCorrector:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Коррекция дисторсии")
        
        # Загрузка изображения
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError("Не удалось загрузить изображение")
        
        self.h, self.w = self.original_image.shape[:2]
        
        # Параметры дисторсии
        self.k1 = tk.DoubleVar(value=0.0)
        self.k2 = tk.DoubleVar(value=0.0)
        self.k3 = tk.DoubleVar(value=0.0)
        self.p1 = tk.DoubleVar(value=0.0)
        self.p2 = tk.DoubleVar(value=0.0)
        
        self.setup_ui()
        self.update_image()
    
    def setup_ui(self):
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Изображение
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=0, column=0, columnspan=6, padx=5, pady=5)
        
        # Слайдеры для параметров дисторсии
        sliders = [
            ("k1 (Радиальная дисторсия 1):", self.k1, -0.5, 0.5),
            ("k2 (Радиальная дисторсия 2):", self.k2, -0.5, 0.5),
            ("k3 (Радиальная дисторсия 3):", self.k3, -0.1, 0.1),
            ("p1 (Тангенциальная дисторсия 1):", self.p1, -0.1, 0.1),
            ("p2 (Тангенциальная дисторсия 2):", self.p2, -0.1, 0.1)
        ]
        
        for i, (text, var, min_val, max_val) in enumerate(sliders, 1):
            label = ttk.Label(main_frame, text=text)
            label.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            
            scale = ttk.Scale(main_frame, from_=min_val, to=max_val, 
                             variable=var, orient=tk.HORIZONTAL, length=200)
            scale.grid(row=i, column=1, columnspan=3, padx=5, pady=2)
            scale.bind("<Motion>", lambda e: self.update_image())
            
            value_label = ttk.Label(main_frame, textvariable=var)
            value_label.grid(row=i, column=4, padx=5, pady=2)
        
        # Кнопка сброса
        reset_btn = ttk.Button(main_frame, text="Сброс", command=self.reset_values)
        reset_btn.grid(row=6, column=0, columnspan=2, pady=10)
        
        # Кнопка сохранения
        save_btn = ttk.Button(main_frame, text="Сохранить параметры", command=self.save_params)
        save_btn.grid(row=6, column=2, columnspan=2, pady=10)
    
    def correct_distortion(self, image, k1, k2, k3, p1, p2):
        """Коррекция дисторсии изображения с сохранением размера"""
        # Матрица камеры
        camera_matrix = np.array([
            [self.w, 0, self.w/2],
            [0, self.h, self.h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Коэффициенты дисторсии
        dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
        
        # Коррекция дисторсии БЕЗ обрезки
        # Используем тот же размер вывода, что и входной
        corrected = cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)
        
        return corrected
    
    def update_image(self, event=None):
        """Обновление изображения с текущими параметрами"""
        k1 = self.k1.get()
        k2 = self.k2.get()
        k3 = self.k3.get()
        p1 = self.p1.get()
        p2 = self.p2.get()
        
        corrected = self.correct_distortion(self.original_image.copy(), k1, k2, k3, p1, p2)
        
        # Конвертация для Tkinter
        corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(corrected_rgb)
        
        # Масштабирование для отображения (сохраняем пропорции)
        max_size = 800
        if max(pil_image.size) > max_size:
            ratio = max_size / max(pil_image.size)
            new_size = tuple(int(dim * ratio) for dim in pil_image.size)
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        tk_image = ImageTk.PhotoImage(pil_image)
        
        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image
    
    def reset_values(self):
        """Сброс параметров к нулю"""
        self.k1.set(0.0)
        self.k2.set(0.0)
        self.k3.set(0.0)
        self.p1.set(0.0)
        self.p2.set(0.0)
        self.update_image()
    
    def save_params(self):
        """Сохранение параметров дисторсии"""
        params = {
            'k1': self.k1.get(),
            'k2': self.k2.get(),
            'k3': self.k3.get(),
            'p1': self.p1.get(),
            'p2': self.p2.get(),
            'image_size': [self.w, self.h]
        }
        
        import json
        with open('distortion_params.json', 'w') as f:
            json.dump(params, f, indent=4)
        
        print("Параметры сохранены в distortion_params.json")

def main():
    # Укажите путь к вашему изображению
    image_path = "CalibParams/photo_0000_20250913_191411.jpg"
    
    root = tk.Tk()
    app = DistortionCorrector(root, image_path)
    root.mainloop()

if __name__ == "__main__":
    main()
