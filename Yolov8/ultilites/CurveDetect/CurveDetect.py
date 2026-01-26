import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import math


class DistanceMeasureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Измерение расстояния между точками")

        # Настройки холста
        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        # Переменные для хранения данных
        self.line_points = []
        self.measure_points = []
        self.background_image = None
        self.tk_image = None

        # Кнопки управления
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=10)

        self.load_btn = tk.Button(self.btn_frame, text="Загрузить изображение", command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.line_btn = tk.Button(self.btn_frame, text="Рисовать линию", command=self.start_drawing_line)
        self.line_btn.pack(side=tk.LEFT, padx=5)

        self.measure_btn = tk.Button(self.btn_frame, text="Измерить расстояние", command=self.start_measuring)
        self.measure_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(self.btn_frame, text="Очистить", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Метка для отображения расстояния
        self.distance_label = tk.Label(root, text="Расстояние: -", font=('Arial', 14))
        self.distance_label.pack()

        # Текущий режим (0 - ничего, 1 - рисование линии, 2 - измерение расстояния)
        self.mode = 0

        # Привязка событий мыши
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            try:
                self.background_image = Image.open(file_path)
                self.background_image = self.background_image.resize((self.canvas_width, self.canvas_height),
                                                                     Image.LANCZOS)
                self.tk_image = ImageTk.PhotoImage(self.background_image)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
                self.clear_canvas(keep_image=True)
            except Exception as e:
                print(f"Ошибка загрузки изображения: {e}")

    def start_drawing_line(self):
        self.mode = 1
        self.line_points = []
        self.distance_label.config(text="Режим: Рисование линии. Кликайте по холсту для создания точек.")

    def start_measuring(self):
        self.mode = 2
        self.measure_points = []
        self.distance_label.config(text="Режим: Измерение расстояния. Кликните две точки.")

    def clear_canvas(self, keep_image=False):
        self.canvas.delete("all")
        if keep_image and self.tk_image:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.line_points = []
        self.measure_points = []
        self.mode = 0
        self.distance_label.config(text="Расстояние: -")

    def on_canvas_click(self, event):
        if self.mode == 1:  # Режим рисования линии
            self.line_points.append((event.x, event.y))
            self.draw_line()
        elif self.mode == 2:  # Режим измерения расстояния
            if len(self.measure_points) < 2:
                self.measure_points.append((event.x, event.y))
                self.draw_measure_points()
                if len(self.measure_points) == 2:
                    distance = self.calculate_distance()
                    self.distance_label.config(text=f"Расстояние: {distance:.2f} пикселей")

    def draw_line(self):
        self.canvas.delete("line")
        if len(self.line_points) > 1:
            self.canvas.create_line(self.line_points, fill="blue", width=2, tags="line")
        for point in self.line_points:
            self.canvas.create_oval(point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3,
                                    fill="red", outline="red", tags="line")

    def draw_measure_points(self):
        self.canvas.delete("measure")
        for point in self.measure_points:
            self.canvas.create_oval(point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5,
                                    fill="green", outline="green", tags="measure")
        if len(self.measure_points) == 2:
            self.canvas.create_line(self.measure_points, fill="red", width=2, tags="measure")

    def calculate_distance(self):
        if len(self.measure_points) == 2:
            x1, y1 = self.measure_points[0]
            x2, y2 = self.measure_points[1]
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return 0


if __name__ == "__main__":
    root = tk.Tk()
    app = DistanceMeasureApp(root)
    root.mainloop()