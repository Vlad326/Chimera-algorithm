import math
import statistics
import traceback
import cv2
import keyboard
import numpy as np
from ultralytics import YOLO
import time
import os
import json
import torch
import Yolov8.DetectionOBB
import ArucoMarkers.Detection
import Yolov8.DetectionOBBOrientation
import Network.СlientToXimera
import matplotlib.pyplot as plt
from functools import cache
import torchvision.transforms as transforms
from PIL import Image
import cProfile
import pstats
import io
import Network.CameraNetwork


def get_dist(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def find_direction(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    angle = math.degrees(math.atan2(dy, dx)) % 360
    return angle


def DrawDirectionArrow(img, center, angle_deg, length=50, color=(0, 255, 0), thickness=3):
    """Рисует стрелку направления на изображении"""
    angle_rad = np.deg2rad(angle_deg)
    end_point = (
        int(center[0] + length * np.cos(angle_rad)),
        int(center[1] + length * np.sin(angle_rad))
    )
    cv2.arrowedLine(img, center, end_point, color, thickness, tipLength=0.3)


def LoadCalibrationData(calib_file='Calibration_camera/calib_param.json'):
    """Загружает калибровочные данные из JSON файла"""
    try:
        with open(calib_file, 'r') as f:
            calib_data = json.load(f)
        print(f"Calibration data loaded from {calib_file}")
        return calib_data
    except FileNotFoundError:
        print(f"Calibration file {calib_file} not found. Using default values.")
        return None
    except json.JSONDecodeError:
        print(f"Error reading file {calib_file}. Using default values")
        return None


@cache
def normalize_angle(angle_deg):
    """Нормализует угол в диапазон [0, 360)"""
    angle_deg = angle_deg % 360
    return angle_deg if angle_deg >= 0 else angle_deg + 360


class FastYOLO:
    def __init__(self, model):
        self.model = model
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.thread = Thread(target=self.process, daemon=True)
        self.thread.start()

    def process(self):
        while True:
            frame = self.frame_queue.get()
            with torch.no_grad():
                results = self.model(frame, imgsz=[1280, 736], conf=0.8, verbose=False)
            self.result_queue.put(results)

    def predict(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame.copy())
        return not self.result_queue.empty()

    def get_result(self):
        return self.result_queue.get() if not self.result_queue.empty() else None


def crop_obb_with_padding(image, obb, padding=5):
    """
    Вырезает OBB с padding, но без поворота (использует AABB).
    """
    # 1. Получаем все X и Y координаты OBB
    xs = [p[0] for p in obb]
    ys = [p[1] for p in obb]

    # 2. Находим AABB (Axis-Aligned Bounding Box) с padding
    x_min = max(0, int(min(xs) - padding))
    y_min = max(0, int(min(ys) - padding))
    x_max = min(image.shape[1], int(max(xs) + padding))
    y_max = min(image.shape[0], int(max(ys) + padding))

    # 3. Проверяем, что регион валиден
    if x_max <= x_min or y_max <= y_min:
        return None

    # 4. Вырезаем прямоугольник
    cropped = image[y_min:y_max, x_min:x_max]

    return cropped


def GetOrientation(rectangle_corners):
    p1, p2, p3, p4 = rectangle_corners

    # Находим центр прямоугольника
    center = np.mean(rectangle_corners, axis=0)

    # Проверяем, какие грани длинные (сравниваем длины противоположных сторон)
    edge1_len = np.linalg.norm(np.array(p2) - np.array(p1))
    edge2_len = np.linalg.norm(np.array(p4) - np.array(p1))

    if edge1_len > edge2_len:
        long_edge1 = (p1, p2)  # Первая длинная грань
        long_edge2 = (p3, p4)  # Вторая длинная грань
    else:
        long_edge1 = (p1, p4)  # Первая длинная грань
        long_edge2 = (p2, p3)  # Вторая длинная грань

    # Находим середины длинных граней
    mid1 = np.mean(long_edge1, axis=0)
    mid2 = np.mean(long_edge2, axis=0)
    # Вычисляем векторы направлений к центру
    vec1 = center - mid1
    vec2 = center - mid2

    # Переводим векторы в углы (в градусах)
    angle1 = np.degrees(np.arctan2(vec1[1], vec1[0])) % 360  # Угол от 0° до 360°
    angle2 = np.degrees(np.arctan2(vec2[1], vec2[0])) % 360
    main_angle = normalize_angle(np.mean([angle1, angle2]) + 90)

    # angle1 = main_angle
    angles = []

    angles.append(main_angle)
    angles.append(normalize_angle(main_angle) + 90)
    angles.append(normalize_angle(main_angle) + 180)
    angles.append(normalize_angle(main_angle) + 270)

    return angles


def GetAlhgoritmOrientation(old_angle, obb):
    angles = GetOrientation(obb)

    angles_r = []
    if old_angle:
        for i in angles:
            angles_r.append(max([old_angle, i]) - min(old_angle, i))

        if min(angles_r) < 70:
            angle = angles[angles_r.index(min(angles_r))]
        else:
            angle = old_angle
    else:
        angle = angles[1]

    return angle


def setup_fp16_models():
    """Настройка моделей для работы с FP16"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Проверяем поддержку FP16 на GTX 1650
    if device == 'cuda':
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] >= 7:  # Volta и новее
            print("FP16 полностью поддерживается")
            fp16_enabled = True
        elif compute_capability[0] >= 6:  # Pascal (GTX 1650)
            print("FP16 поддерживается с ограничениями")
            fp16_enabled = True
        else:
            print("FP16 не поддерживается, используем FP32")
            fp16_enabled = False
    else:
        fp16_enabled = False

    return device, fp16_enabled


# Функция для рисования черного прямоугольника вокруг точки
def draw_black_rectangle_around_point(img, point, rect_size=(80, 60)):
    """
    Рисует черный прямоугольник вокруг указанной точки
    """
    x, y = point
    width, height = rect_size

    # Вычисляем координаты верхнего левого и нижнего правого углов
    x1 = max(0, x - width // 2)
    y1 = max(0, y - height // 2)
    x2 = min(img.shape[1], x + width // 2)
    y2 = min(img.shape[0], y + height // 2)

    # Рисуем черный прямоугольник
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)  # -1 означает заливку
    return img

def main_profiled():
    # Настройка оптимизаций для GPU
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.empty_cache()

    # Настройка FP16
    device, fp16_enabled = setup_fp16_models()

    RingWidth = 5000  # мм
    RingHeight = 5000  # мм

    print("INFO:")

    AlghoritmOrientation = True
    YoloOrientation = False
    # если включить оба то будет гибридный режим

    CalibrationCamera = "C:/Users/User/PycharmProjects/PythonProject/Project/algoritmXimeraV111/CalibrationCamera/CalibParamsOuts/CalibParams_22.json"
    YoloModelPath = "C:/Users/User/PycharmProjects/PythonProject/Project/algoritmXimeraV1/Yolov8/ultilites/JustTrain/TrainingYoloOBBModel/runs/obb/yolov8n_robot2_obb12/weights/best.pt"
    YoloModelOrientationPath = "C:/Users/User/PycharmProjects/PythonProject/Project/algoritmXimeraV1/Yolov8/ultilites/JustTrain/TrainingYoloOBBModel/runs/obb/yolov8n_robot2_obb14/weights/best.pt"

    """
    RUBEZH
    YoloModelPath = "C:/Users/User/PycharmProjects/PythonProject/Project/algoritmXimeraV1/Yolov8/ultilites/JustTrain/TrainingYoloOBBModel/runs/obb/yolov8n_robot2_obb10/weights/best.pt"
    TANTAL
    YoloModelPath = "C:/Users/User/PycharmProjects/PythonProject/Project/algoritmXimeraV1/Yolov8/ultilites/JustTrain/TrainingYoloOBBModel/runs/obb/yolov8n_robot2_obb9/weights/best.pt"
    Wolfram
    YoloModelPath = "C:/Users/User/PycharmProjects/PythonProject/Project/algoritmXimeraV1/Yolov8/ultilites/JustTrain/TrainingYoloOBBModel/runs/obb/yolov8n_robot2_obb12/weights/best.pt"
    
    """
    #CalibrationCamera = "CalibrationCamera/CalibParamsOuts/CalibParams_40.json"
    # ArucoMarkers
    markers_ids = {
        "marker_up_DownSide": 1,
        "marker_down_DownSide": 13,
        "marker_right_DownSide": 2,
        "marker_left_DownSide": 4,
        "marker_up_UpSide": 5,
        "marker_down_UpSide": 7,
        "marker_right_UpSide": 6,
        "marker_left_UpSide": 8
    }  # id каждого аруго маркера

    markers_None = {
        "marker_up_DownSide": -1,
        "marker_down_DownSide": -1,
        "marker_right_DownSide": -1,
        "marker_left_DownSide": -1,
        "marker_up_UpSide": -1,
        "marker_down_UpSide": -1,
        "marker_right_UpSide": -1,
        "marker_left_UpSide": -1
    }  # значение аруго маркеров по умолчанию

    markers_off = {
        "marker_up_DownSide": -90,
        "marker_down_DownSide": 90,
        "marker_right_DownSide": 0,
        "marker_left_DownSide": 180,
        "marker_up_UpSide": -90,
        "marker_down_UpSide": 90,
        "marker_right_UpSide": 0,
        "marker_left_UpSide": 180
    }  # смещение при повороте кажого аруго маркера относительно линии прямой робота

    print(f"Using device: {device.upper()} for Yolo")
    if fp16_enabled:
        print("FP16 optimization: ENABLED")
    else:
        print("FP16 optimization: DISABLED")

    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("GPU available for opencv")
    else:
        print("CPU available for opencv")

    calib_data = LoadCalibrationData(CalibrationCamera)

    # Загрузка моделей с FP16
    YoloModel = YOLO(YoloModelPath)
    YoloModelDirection = YOLO(YoloModelOrientationPath)

    YoloModel.to('cuda')
    # Конвертация моделей в FP16 если поддерживается
    if fp16_enabled:
        try:
            # Конвертируем модели в FP16
            YoloModel.model.half()
            YoloModelDirection.model.half()
            print("Models converted to FP16")
        except Exception as e:
            print(f"Error converting models to FP16: {e}")
            print("Falling back to FP32")
            fp16_enabled = False

    # Перемещаем модели на устройство
    YoloModel.model.to(device)
    YoloModelDirection.model.to(device)

    video_source = 1  # 'Yolov8/video.mp4'      "robots/rubezh/out.avi"
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Ошибка открытия видео источника: {video_source}")
        return

    output_dir = 'outs'
    os.makedirs(output_dir, exist_ok=True)

    #imgsz = [1080, 1920]

    #imgsz = [1536, 2048]

    #imgsz = [1944, 2592]

    #imgsz = [1920,1080]

    #imgsz = [2048,1536]

    imgsz = [2592, 1944]

    SenderToXimera = Network.СlientToXimera.SenderToXimera()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, imgsz[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imgsz[1])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    cap.set(cv2.CAP_PROP_FPS, 10)
    #cap.set(cv2.CAP_PROP_GAIN, 0)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
    cap.set(cv2.CAP_PROP_CONTRAST, 100)
    cap.set(cv2.CAP_PROP_SATURATION, 50)
    """
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
    cap.set(cv2.CAP_PROP_CONTRAST, 100)
    cap.set(cv2.CAP_PROP_SATURATION, 100)
    """
    cap.set(cv2.CAP_PROP_SHARPNESS, 100)

    #cap.set(cv2.CAP_PROP_GAIN, 75)
    cap.set(cv2.CAP_PROP_EXPOSURE, -8)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Requested resolution: {imgsz[0]}x{imgsz[1]}, Actual resolution: {actual_width}x{actual_height}")

    """cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Ручной режим экспозиции
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Уменьшение экспозиции (меньше размытие)
    cap.set(cv2.CAP_PROP_GAIN, 75)  # Увеличение усиления для компенсации
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)  # Средняя яркость
    cap.set(cv2.CAP_PROP_CONTRAST, 70)  # Увеличенная контрастность
    cap.set(cv2.CAP_PROP_SATURATION, 70)  # Увеличенная насыщенность
    cap.set(cv2.CAP_PROP_SHARPNESS, 100)  # Максимальная резкость
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, imgsz[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imgsz[0])
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Минимальный буфер"""

    fps = 30

    base_name = 'outs/out'
    extension = '.avi'
    counter = 1
    output_filename = base_name + extension

    while os.path.exists(output_filename):
        output_filename = f"{base_name}_{counter}{extension}"
        counter += 1

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, 8, (actual_width, actual_height))

    prev_time = 0
    fps_values = []

    if calib_data and 'K' in calib_data and 'D' in calib_data:
        camera_matrix = np.array(calib_data['K'])
        dist_coeffs = np.array(calib_data['D'][0])
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (actual_width, actual_height), 1, (actual_width, actual_height))
        print("Distortion correction will be applied")
    else:
        camera_matrix = None
        print("Distortion correction will not be applied")

    ArucoDetector = ArucoMarkers.Detection.setup_detection(markers_ids, markers_None)

    old_angle = None
    frame_counter = 0
    saved_count = 0

    print()
    print("-" * 10)
    print()

    pix_width = RingWidth / actual_width
    pix_height = RingHeight / actual_height

    timings = {
        'capture': 0,
        'yolo': 0,
        'aruco': 0,
        'visualization': 0,
        'total': 0
    }

    min_processing_time = 0.033  # Минимальное время между кадрами (~30 FPS)

    DEBUG = True


    try:
        while True:
            # Пропускаем кадры если обработка слишком быстрая
            if time.time() - prev_time < min_processing_time:
                continue

            total_start = time.time()

            capture_start = time.time()

            ret, frame = cap.read()

            # cv2.imshow('frame', frame)
            # frame = frame[133:908, 165:1780]

            timings['capture'] = time.time() - capture_start

            if camera_matrix is not None:
                frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
                frame_without_roi = frame.copy()
                x, y, w, h = roi
                frame = frame[y:y + h, x:x + w] if w > 0 and h > 0 else frame

            # ArUco обработка ПЕРВОЙ
            aruco_start = time.time()
            ArucoMarkersResults = ArucoMarkers.Detection.GetResultsFrame(
                frame, ArucoDetector, markers_ids, markers_None, markers_off, prev_time)
            timings['aruco'] = time.time() - aruco_start

            # Рисуем черный прямоугольник вокруг центра ArUco маркера
            if ArucoMarkersResults['robot']['position'] is not None and len(
                    ArucoMarkersResults['robot']['position']) > 1:
                aruco_center = ArucoMarkersResults['robot']['position']



                # Рисуем черный прямоугольник вокруг центра ArUco
                frame = draw_black_rectangle_around_point(frame, aruco_center, (200, 200))

                if camera_matrix is not None:
                    x, y, w, h = roi
                else:
                    x, y = 0, 0

                ArucoMarkersResults['robot'].update({
                    'RealPosition': [
                        int((ArucoMarkersResults['robot']['position'][0] + x - 520) * pix_width),
                        int((ArucoMarkersResults['robot']['position'][1] + y - 340) * pix_height)
                    ]
                })

            yolo_start = time.time()

            # Конвертируем frame в FP16 если используется
            if fp16_enabled:
                # Для Ultralytics YOLO обычно автоматически обрабатывает типы данных
                # Но мы можем явно указать использовать half precision при inference
                with torch.cuda.amp.autocast(enabled=fp16_enabled):
                    AllResultsFrameYolo = Yolov8.DetectionOBB.GetResultsFrame(YoloModel, frame, imgsz=[1280, 960],
                                                                              conf=0.7)
            else:
                AllResultsFrameYolo = Yolov8.DetectionOBB.GetResultsFrame(YoloModel, frame, imgsz=[1280, 960],
                                                                          conf=0.7)

            results = AllResultsFrameYolo["results"]
            obb = AllResultsFrameYolo["obb"]

            # annotated_frame = results[0].plot()
            annotated_frame = frame.copy()

            RobotOppCenter = [-1, -1]
            RobotOppAngle = -1
            if obb is not None and len(obb) >= 4:
                RobotOppCenter = [
                    int(np.mean([obb[0][0], obb[1][0], obb[2][0], obb[3][0]])),
                    int(np.mean([obb[0][1], obb[1][1], obb[2][1], obb[3][1]]))
                ]

                RobotOppCenterMM = [
                    int(np.mean([obb[0][0], obb[1][0], obb[2][0], obb[3][0]]) * pix_width),
                    int(np.mean([obb[0][1], obb[1][1], obb[2][1], obb[3][1]]) * pix_height)
                ]

                # Рисуем точки углов
                for i, point in enumerate(obb):
                    cv2.circle(annotated_frame, (int(point[0]), int(point[1])), 5, [255, 0, 0], 5)

                if YoloOrientation:
                    cropped_obj = crop_obb_with_padding(frame, obb)
                    if cropped_obj.size > 0:
                        if fp16_enabled:
                            with torch.cuda.amp.autocast(enabled=fp16_enabled):
                                OrientationYoloResults = Yolov8.DetectionOBBOrientation.GetResultsFrameOrientation(
                                    YoloModelDirection, cropped_obj, 0.4, imgsz=416 // 2)
                        else:
                            OrientationYoloResults = Yolov8.DetectionOBBOrientation.GetResultsFrameOrientation(
                                YoloModelDirection, cropped_obj, 0.4, imgsz=416 // 2)

                        DirectionObb = OrientationYoloResults['obb']

                        if DirectionObb is not None:
                            CenterDirectionYoloResultsLocal = [
                                np.mean(
                                    [DirectionObb[0][0], DirectionObb[1][0], DirectionObb[2][0], DirectionObb[3][0]]),
                                np.mean(
                                    [DirectionObb[0][1], DirectionObb[1][1], DirectionObb[2][1], DirectionObb[3][1]])
                            ]
                            CenterDirectionYoloResultsGlobal = [
                                CenterDirectionYoloResultsLocal[0] + RobotOppCenter[0] - cropped_obj.shape[1] // 2,
                                CenterDirectionYoloResultsLocal[1] + RobotOppCenter[1] - cropped_obj.shape[0] // 2]

                            RobotOppAngle = find_direction(RobotOppCenter, CenterDirectionYoloResultsGlobal)
                        elif AlghoritmOrientation != None:
                            RobotOppAngle = old_angle

                if AlghoritmOrientation:
                    if YoloOrientation:
                        if DirectionObb == None:
                            RobotOppAngle = GetAlhgoritmOrientation(old_angle, obb)
                    else:
                        RobotOppAngle = GetAlhgoritmOrientation(old_angle, obb)

                if DEBUG:
                    DrawDirectionArrow(annotated_frame, (int(RobotOppCenter[0]), int(RobotOppCenter[1])), RobotOppAngle,
                                       length=75)

                    cv2.putText(annotated_frame, f"AVG: ({RobotOppCenter[0], RobotOppCenter[1]}), {RobotOppCenterMM}",
                                (RobotOppCenter[0] + 20,
                                 RobotOppCenter[1] + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                old_angle = RobotOppAngle
                frame_counter += 1

            timings['yolo'] = time.time() - yolo_start

            #print(ArucoMarkersResults['robot']['position'])
            #print()
            #print(RobotOppCenter[0])
            #print(RobotOppCenter[1])
            #print()

            if ArucoMarkersResults['robot']['direction'] is not None:
                ArucoMarkersResults['robot']['direction'] = normalize_angle(ArucoMarkersResults['robot']['direction'])

            if ArucoMarkersResults['markers']['corners'] is not None and ArucoMarkersResults['markers'][
                'ids'] is not None and DEBUG:
                cv2.aruco.drawDetectedMarkers(annotated_frame, ArucoMarkersResults['markers']['corners'],
                                              ArucoMarkersResults['markers']['ids'])

            if ArucoMarkersResults['markers']['centers'] is not None and len(
                    ArucoMarkersResults['markers']['centers']) > 0 and DEBUG:
                cv2.putText(annotated_frame, "ROBOT INFO:",
                            (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (127, 217, 102), 2)

            # Side
            if ArucoMarkersResults['robot']['side'] and DEBUG:
                cv2.putText(annotated_frame, "Side:" + ArucoMarkersResults['robot']['side'],
                            (10, 80 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Robot angle
            if ArucoMarkersResults['robot']['direction'] is not None and DEBUG:
                DrawDirectionArrow(annotated_frame,
                                   tuple(ArucoMarkersResults['robot']['position']),
                                   ArucoMarkersResults['robot']['direction'])

                cv2.putText(annotated_frame, f"Dir: {round(ArucoMarkersResults['robot']['direction'], 3)}",
                            (ArucoMarkersResults['robot']['position'][0] + 20,
                             ArucoMarkersResults['robot']['position'][1] + 60 - 29),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.putText(annotated_frame, f"Dir: {round(ArucoMarkersResults['robot']['direction'], 3)}",
                            (10,
                             80 + 30 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if ArucoMarkersResults['robot']['position'] is None:
                ArucoMarkersResults['robot']['position'] = [-1, -1]

            if ArucoMarkersResults['robot']['direction'] is None:
                ArucoMarkersResults['robot']['direction'] = -1

            if ArucoMarkersResults['robot']['side'] == "UpSide":
                ArucoMarkersResults['robot']['side'] = 0
            elif ArucoMarkersResults['robot']['side'] == "DownSide":
                ArucoMarkersResults['robot']['side'] = 1
            else:
                ArucoMarkersResults['robot']['side'] = -1

            SenderToXimera.SendToXimeraData(InputData={
                "Ximera": {
                    "X": ArucoMarkersResults['robot']['position'][0],
                    "Y": ArucoMarkersResults['robot']['position'][1],
                    "R": ArucoMarkersResults['robot']['direction'],
                    "Side": ArucoMarkersResults['robot']['side']
                },
                "Opponent": {
                    "X": RobotOppCenter[0],
                    "Y": RobotOppCenter[1],
                    "R": RobotOppAngle
                }
            })

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            fps_values.append(fps)
            prev_time = current_time

            # cv2.imshow('Window', annotated_frame)
            if DEBUG:

                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # annotated_frame = cv2.resize(annotated_frame, [1280, 720])
                cv2.imshow('Window', annotated_frame)
                if out is not None:
                    if annotated_frame.shape[1] != actual_width or annotated_frame.shape[0] != actual_height:
                        annotated_frame = cv2.resize(annotated_frame, (actual_width, actual_height))

                    try:
                        out.write(annotated_frame)
                    except Exception as e:
                        print(f"Error writing frame to video: {e}")
                        out.release()
                        out = None

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            timings['total'] = time.time() - total_start

            if keyboard.is_pressed('q') or keyboard.is_pressed('Q'):
                break

            # Вывод статистики
            if frame_counter % 30 == 0:
                print(f"YOLO: {timings['yolo'] * 1000:.1f}ms, "
                      f"ArUco: {timings['aruco'] * 1000:.1f}ms, "
                      f"Total: {timings['total'] * 1000:.1f}ms")

                if fp16_enabled:
                    memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
                    memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2
                    print(f"GPU Memory: {memory_allocated:.1f}MB allocated, {memory_reserved:.1f}MB reserved")

    except Exception as e:
        print("Error:")
        traceback.print_exc()
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

        if fps_values:
            min_fps = min(fps_values)
            avg_fps = statistics.mean(fps_values)
            max_fps = max(fps_values)

            print("\nFPS Statistics:")
            print(f"Minimum FPS: {min_fps:.3f}")
            print(f"Average FPS: {avg_fps:.3f}")
            print(f"Maximum FPS: {max_fps:.3f}")
            print(f"Total frames processed: {len(fps_values)}")

        x = [i for i in range(0, len(fps_values))]

        # Create a simple line plot
        plt.plot(x, fps_values)
        plt.title("FPS Statistics")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()


def main():
    main_profiled()


if __name__ == "__main__":
    # Убедитесь что все операции на GPU
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)  # Отключаем градиенты для инференса

    main()
