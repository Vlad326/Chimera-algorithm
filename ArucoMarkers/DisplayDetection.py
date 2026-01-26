import json
import os
import cv2
import numpy as np
import time
import math
import statistics

def load_calibration_data(calib_file="calib_param_single.json"):
    """Загружает данные калибровки из JSON-файла"""
    try:
        with open(calib_file, "r") as f:
            calib_data = json.load(f)
        
        K = np.array(calib_data["K"])
        D = np.array(calib_data["D"])
        imSize = tuple(calib_data["imSize"])
        
        # Оптимизируем матрицу камеры
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, imSize, 1, imSize)
        
        return {
            "K": K,
            "D": D,
            "new_K": new_K,
            "roi": roi,
            "imSize": imSize
        }
    except Exception as e:
        print(f"Ошибка загрузки калибровочных данных: {e}")
        return None

def setup_detection(markers_id, markers_None):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.01
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    return detector

def get_next_video_filename():
    if not os.path.exists('outs'):
        os.makedirs('outs')
    i = 1
    while True:
        filename = f'outs/out_{i}.avi'
        if not os.path.exists(filename):
            return filename
        i += 1

def draw_line_direction(img, center, angle_rad, length=50, color=(0, 255, 0), thickness=3):
    direction = (math.cos(angle_rad), math.sin(angle_rad))
    end_point = (int(center[0] + direction[0] * length),
                 int(center[1] + direction[1] * length))
    cv2.line(img, center, end_point, color, thickness)
    cv2.circle(img, end_point, 5, color, -1)
    

def filter_selected_markers(frame, corners, ids, markers_id, markers_None):
    """
    Filters markers to keep only those that are in the selected markers_None list.
    Draws rejected markers with red outlines.
    """
    if ids is None:
        return corners, ids

    # Get all valid marker IDs from the markers_None list
    valid_ids = [markers_id[name] for name in markers_None if name in markers_id]

    # Find indices of markers to keep and reject
    keep_indices = []
    reject_indices = []

    for i, id_num in enumerate(ids.flatten()):
        if id_num in valid_ids:
            keep_indices.append(i)
        else:
            reject_indices.append(i)

    # Draw rejected markers in red
    for i in reject_indices:
        corner = corners[i][0].astype(int)
        cv2.polylines(frame, [corner], True, (0, 0, 255), 2)
        center_x = int(corner[:, 0].mean())
        center_y = int(corner[:, 1].mean())
        cv2.putText(frame, f"Rejected ID: {ids[i][0]}", (center_x + 15, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Filter the markers
    if len(keep_indices) < len(ids):
        ids = ids[keep_indices]
        corners = [corners[i] for i in keep_indices]

    return corners, ids


def rotate_angle(current_angle, change_degrees, angle_range='-180-180'):
    """
    Updates a direction angle with smooth rotation within bounds.
    Integrates with your existing avg_angle calculation.
    """
    # Convert current angle to degrees
    current_deg = math.degrees(current_angle)

    # Calculate new angle in degrees
    new_deg = (current_deg + change_degrees) % 360

    # Convert to -180-180 range if specified
    if angle_range == '-180-180':
        if new_deg > 180:
            new_deg -= 360

    # Convert back to radians for consistency with your code
    new_rad = math.radians(new_deg)

    return new_rad, new_deg


def calc_direction_2(a, b):
    if a is not None and b is not None:
        vec = (a[0] - b[0], a[1] - b[1])
        angle_ver = math.atan2(vec[1], vec[0])
        return angle_ver


def calculate_direction(markers_centers, robot_DownSide):
    direction_angle = None
    direction_text = "UNKNOWN"

    if robot_DownSide:
        up = markers_centers.get('marker_up_DownSide')
        down = markers_centers.get('marker_down_DownSide')
        left = markers_centers.get('marker_left_DownSide')
        right = markers_centers.get('marker_right_DownSide')
    else:
        up = markers_centers.get('marker_up_UpSide')
        down = markers_centers.get('marker_down_UpSide')
        left = markers_centers.get('marker_left_UpSide')
        right = markers_centers.get('marker_right_UpSide')

    angles = []

    angle_ver = calc_direction_2(up, down)
    if angle_ver:
        angles.append(angle_ver)

    angle_ver = calc_direction_2(up, right)
    if angle_ver:
        angle_ver = rotate_angle(angle_ver, 45)[0]
        angles.append(angle_ver)

    angle_ver = calc_direction_2(up, left)
    if angle_ver:
        angle_ver = rotate_angle(angle_ver, -45)[0]
        angles.append(angle_ver)

    angle_ver = calc_direction_2(right, left)
    if angle_ver:
        angle_ver = rotate_angle(angle_ver, -90)[0]
        angles.append(angle_ver)

    angle_ver = calc_direction_2(right, down)
    if angle_ver:
        angle_ver = rotate_angle(angle_ver, -45)[0]
        angles.append(angle_ver)

    angle_ver = calc_direction_2(left, down)
    if angle_ver:
        angle_ver = rotate_angle(angle_ver, 45)[0]
        angles.append(angle_ver)

    if angles:
        if len(angles) == 2:
            avg_angle = np.mean(angles)
        else:
            avg_angle = angles[0]

        angle_deg = math.degrees(avg_angle)
        direction_angle = avg_angle

        angle_deg = angle_deg % 360
        if angle_deg > 180:
            angle_deg -= 360

        if -45 <= angle_deg < 45:
            direction_text = "RIGHT"
        elif 45 <= angle_deg < 135:
            direction_text = "DOWN"
        elif -135 <= angle_deg < -45:
            direction_text = "UP"
        else:
            direction_text = "LEFT"

    return direction_angle, direction_text


def filter_opposite_markers(frame, corners, ids, markers_id, markers_None):
    """
    When three markers are detected (two opposite and one non-opposite), calculates the center point
    by drawing a perpendicular line from the non-opposite marker to the line between the opposite markers.
    Returns:
        corners: Filtered corners
        ids: Filtered ids
        centers: List of remaining centers in format [(x1, y1), (x2, y2), ...]
        avg_center: Calculated center point (None if not calculated)
    """
    centers = []  # List to store centers of remaining markers
    avg_center = None  # Calculated center point

    if ids is None:
        return corners, ids, centers, avg_center

    # Calculate centers for all markers
    centers = [tuple(map(int, corner[0].mean(axis=0))) for corner in corners]

    if len(ids) != 3:
        return corners, ids, centers, avg_center

    detected_markers = []
    for id_num in ids.flatten():
        for marker_name, marker_id in markers_id.items():
            if id_num == marker_id:
                detected_markers.append(marker_name)
                break

    sides = [name.split('_')[-1] for name in detected_markers]
    if len(set(sides)) != 1:
        return corners, ids, centers, avg_center

    directions = [name.split('_')[1] for name in detected_markers]
    opposite_pairs = [('up', 'down'), ('left', 'right')]

    for pair in opposite_pairs:
        if pair[0] in directions and pair[1] in directions:
            # Find the indices of the opposite markers and the third marker
            opposite_indices = []
            third_marker_index = None
            for i, direction in enumerate(directions):
                if direction in pair:
                    opposite_indices.append(i)
                else:
                    third_marker_index = i

            if len(opposite_indices) == 2 and third_marker_index is not None:
                # Get the centers of the opposite markers and the third marker
                p1 = np.array(centers[opposite_indices[0]])
                p2 = np.array(centers[opposite_indices[1]])
                p3 = np.array(centers[third_marker_index])

                # Calculate the line between opposite markers (p1 to p2)
                line_vec = p2 - p1
                line_len = np.linalg.norm(line_vec)
                line_unitvec = line_vec / line_len

                # Calculate the vector from p1 to p3
                p1_to_p3 = p3 - p1

                # Project p1_to_p3 onto the line_vec to find the closest point on the line to p3
                projection_length = np.dot(p1_to_p3, line_unitvec)
                closest_point = p1 + projection_length * line_unitvec

                # Calculate the perpendicular vector from p3 to the line
                perpendicular_vec = closest_point - p3
                perpendicular_len = np.linalg.norm(perpendicular_vec)

                if perpendicular_len > 0:
                    perpendicular_unitvec = perpendicular_vec / perpendicular_len
                else:
                    perpendicular_unitvec = np.array([-line_unitvec[1], line_unitvec[0]])

                # The center is the intersection point (closest_point)
                avg_center = tuple(map(int, closest_point))

                # Draw the lines for visualization
                cv2.line(frame, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 255, 255), 2)
                cv2.line(frame, tuple(p3.astype(int)), tuple(closest_point.astype(int)), (255, 0, 255), 2)
                cv2.circle(frame, tuple(closest_point.astype(int)), 8, (0, 255, 0), -1)

                # Draw the markers
                cv2.circle(frame, tuple(p1.astype(int)), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(p2.astype(int)), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(p3.astype(int)), 5, (255, 0, 0), -1)

                # Label the points
                cv2.putText(frame, "Opposite 1", tuple(p1.astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, "Opposite 2", tuple(p2.astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, "Third Marker", tuple(p3.astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, "Calculated Center", tuple(closest_point.astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                break

    return corners, ids, centers, avg_center


def line_intersection(line1, line2):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

    if denom == 0:
        return None

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    if 0 <= ua <= 1 and 0 <= ub <= 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (int(x), int(y))

    return None


def get_marker_direction(marker_name, direction_angle):
    direction_part = marker_name.split('_')[1]

    if direction_angle is None:
        direction_angle = 0

    direction_angle = math.degrees(direction_angle)

    if direction_part == 'up':
        return math.radians(direction_angle + 180)
    elif direction_part == 'down':
        return math.radians(direction_angle)
    elif direction_part == 'left':
        return math.radians(direction_angle + 90)
    elif direction_part == 'right':
        return math.radians(direction_angle - 90)
    else:
        return math.radians(0)


def find_center_from_two_non_opposite_markers(frame, markers_centers, direction_angle):
    detected_markers = [name for name, center in markers_centers.items() if center is not None]

    if len(detected_markers) != 2:
        return None

    marker1, marker2 = detected_markers
    dir1 = marker1.split('_')[1]
    dir2 = marker2.split('_')[1]

    opposite_pairs = [('up', 'down'), ('down', 'up'),
                      ('left', 'right'), ('right', 'left')]

    if (dir1, dir2) in opposite_pairs:
        return None

    center1 = markers_centers[marker1]
    center2 = markers_centers[marker2]

    direction1 = get_marker_direction(marker1, direction_angle)
    direction2 = get_marker_direction(marker2, direction_angle)

    length = 1000  # Большая длина для "бесконечной" линии

    line1_p1 = (
        int(center1[0] - length * math.cos(direction1)),
        int(center1[1] - length * math.sin(direction1))
    )
    line1_p2 = (
        int(center1[0] + length * math.cos(direction1)),
        int(center1[1] + length * math.sin(direction1))
    )

    line2_p1 = (
        int(center2[0] - length * math.cos(direction2)),
        int(center2[1] - length * math.sin(direction2))
    )
    line2_p2 = (
        int(center2[0] + length * math.cos(direction2)),
        int(center2[1] + length * math.sin(direction2))
    )

    # Рисуем только видимую часть линий (100 пикселей)
    visible_length = 100
    line1_visible_end = (
        int(center1[0] + visible_length * math.cos(direction1)),
        int(center1[1] + visible_length * math.sin(direction1))
    )
    line2_visible_end = (
        int(center2[0] + visible_length * math.cos(direction2)),
        int(center2[1] + visible_length * math.sin(direction2))
    )

    cv2.line(frame, center1, line1_visible_end, (255, 0, 255), 2)
    cv2.line(frame, center2, line2_visible_end, (255, 0, 255), 2)

    intersection = line_intersection((line1_p1, line1_p2), (line2_p1, line2_p2))

    if intersection:
        cv2.circle(frame, intersection, 8, (0, 255, 255), -1)
        return intersection

    return None


def calculate_marker_direction(corners):
    """
    Calculate marker direction based on its corners.
    Returns the angle in radians pointing from the center to the top side of the marker.
    """
    # Convert corners to numpy array (4 points)
    pts = corners[0].astype(np.float32)

    # Find the top-left corner (smallest x + y)
    sum_pts = pts.sum(axis=1)
    top_left_idx = np.argmin(sum_pts)

    # Get the two adjacent corners
    prev_idx = (top_left_idx - 1) % 4
    next_idx = (top_left_idx + 1) % 4

    # Calculate vectors to adjacent corners
    vec1 = pts[next_idx] - pts[top_left_idx]
    vec2 = pts[prev_idx] - pts[top_left_idx]

    # The longer vector is the direction of the marker's side
    if np.linalg.norm(vec1) > np.linalg.norm(vec2):
        direction_vec = vec1
    else:
        direction_vec = vec2

    # Calculate the angle of the direction vector
    angle = math.atan2(direction_vec[1], direction_vec[0])

    # The marker's "front" is perpendicular to its side
    marker_direction = angle + math.pi / 2

    return marker_direction


def calculate_center_from_single_marker(frame, marker_corners, marker_center, marker_name, markers_dir):
    """
    Calculate a point N marker lengths away from the marker in its direction.
    The direction is determined by the marker's orientation plus predefined offset.

    Args:
        frame: Image frame for drawing
        marker_corners: Detected marker corners (numpy array of shape (1,4,2))
        marker_center: Pre-calculated (x,y) center of the marker
        marker_name: ID/name of the marker
        markers_dir: Dictionary of direction offsets for specific markers

    Returns:
        tuple: Calculated (x, y) point and final direction angle in radians
    """
    # Extract corner points
    corners = marker_corners[0]

    # Calculate average marker side length
    def calc_distance(p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    # Calculate lengths of all sides
    side_lengths = [
        calc_distance(corners[0], corners[1]),  # top_right to bottom_right
        calc_distance(corners[1], corners[2]),  # bottom_right to bottom_left
        calc_distance(corners[2], corners[3]),  # bottom_left to top_left
        calc_distance(corners[3], corners[0])  # top_left to top_right
    ]

    # Get average side length
    average_side_length = sum(side_lengths) / len(side_lengths)

    # Calculate orientation vector using both top corners
    top_center = ((corners[3][0] + corners[0][0]) / 2,
                  (corners[3][1] + corners[0][1]) / 2)

    # Create direction vector from center to top center
    dx = top_center[0] - marker_center[0]
    dy = top_center[1] - marker_center[1]

    # Calculate base direction angle (in radians)
    marker_direction = math.atan2(dy, dx)

    # Apply direction offset if specified
    if marker_name in markers_dir:
        direction_angle = marker_direction + math.radians(markers_dir[marker_name])
    else:
        direction_angle = marker_direction

    # Calculate point 1.5 marker lengths away (default distance)
    distance_in_pixels = 2 * average_side_length
    x = int(marker_center[0] + distance_in_pixels * math.cos(direction_angle))
    y = int(marker_center[1] + distance_in_pixels * math.sin(direction_angle))

    # Convert corners to int for drawing
    corners_int = marker_corners[0].astype(int)
    top_right = corners_int[0]
    bottom_right = corners_int[1]
    bottom_left = corners_int[2]
    top_left = corners_int[3]

    # Visualization for debugging
    # 1. Draw main direction line (yellow)
    cv2.line(frame, marker_center, (x, y), (0, 255, 255), 2)
    # 2. Draw target point (green circle)
    cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
    # 3. Draw orientation vector (red line)
    cv2.line(frame, marker_center, tuple(map(int, top_center)), (0, 0, 255), 2)
    # 4. Label corners for verification
    for i, corner in enumerate([top_right, bottom_right, bottom_left, top_left]):
        cv2.putText(frame, str(i), tuple(corner),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return (x, y), direction_angle

def process_frame(frame, detector, markers_id, markers_None, markers_dir, prev_time, fps_history, calib_data):
    # 1. Покажите оригинальное изображение
    #cv2.imshow('Original', frame)
    
    if calib_data:
        # 2. Покажите изображение ДО обрезки ROI
        undistorted = cv2.undistort(frame, calib_data["K"], calib_data["D"], None, calib_data["new_K"])
    
        # 3. Примените ROI
        x, y, w, h = calib_data["roi"]
        undistorted = undistorted[y:y+h, x:x+w]
        frame = undistorted

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #---
    gray = cv2.bitwise_not(gray)
    cv2.imshow(gray)
    #gray=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)
    #---
    frame_with_markers = frame.copy()
    
    # Detect all markers
    corners, ids, rejected = detector.detectMarkers(gray)

    # First filter - outline and remove non-selected markers
    corners, ids = filter_selected_markers(frame_with_markers, corners, ids, markers_id, markers_None)

    markers_centers = {k: None for k in markers_None}
    all_centers = []
    robot_detected_DownSide = False
    single_marker_direction = None

    if ids is not None:
        ids_detect = ids.flatten()

        for id_number in range(len(ids_detect)):
            id_ = ids_detect[id_number]
            for marker_name in markers_None:
                if id_ == markers_id[marker_name]:
                    corner = corners[id_number][0]
                    center_x = int(corner[:, 0].mean())
                    center_y = int(corner[:, 1].mean())
                    markers_centers[marker_name] = (center_x, center_y)
                    all_centers.append((center_x, center_y))

                    if marker_name.endswith("DownSide"):
                        robot_detected_DownSide = True

        # Draw accepted markers
        cv2.aruco.drawDetectedMarkers(frame_with_markers, corners, ids)

        for center in all_centers:
            cv2.circle(frame_with_markers, center, 5, (0, 0, 255), -1)

        for marker_name, center in markers_centers.items():
            if center is not None:
                text = f"{marker_name}: ({center[0]}, {center[1]})"
                cv2.putText(frame_with_markers, text, (center[0], center[1] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(frame_with_markers, "ROBOT INFO:",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (127, 217, 102), 2)

        if len(all_centers) > 0:
            # Get the filtered markers and calculated center (if applicable)
            filtered_corners, filtered_ids, filtered_centers, calculated_center = filter_opposite_markers(
                frame_with_markers, corners, ids, markers_id, markers_None)

            direction_angle, direction_text = calculate_direction(markers_centers, robot_detected_DownSide)

            # Определяем центр в зависимости от количества маркеров
            if len(all_centers) == 1:
                # For single marker, calculate point 100 pixels away in its direction
                marker_idx = 0
                marker_corners = corners[marker_idx]
                marker_center = all_centers[marker_idx]

                # Find the marker name for this ID
                marker_id = ids[marker_idx][0]
                marker_name = None
                for name, id_val in markers_id.items():
                    if id_val == marker_id:
                        marker_name = name
                        break

                if marker_name:
                    avg_center, single_marker_direction = calculate_center_from_single_marker(
                        frame_with_markers, marker_corners, marker_center, marker_name, markers_dir)

                    # Применяем поворот направления в зависимости от типа маркера
                    marker_type = marker_name.split('_')[1]  # Извлекаем тип маркера (up/down/left/right)

                    if marker_type == 'up':
                        # Поворот на 90° вправо
                        direction_angle = single_marker_direction + math.radians(180)
                    elif marker_type == 'down':
                        # Поворот на -90° (90° влево)
                        direction_angle = single_marker_direction + math.radians(0)
                    elif marker_type == 'right':
                        # Поворот на 180°
                        direction_angle = single_marker_direction + math.radians(90)
                    else:  # left и другие случаи
                        # Без изменений
                        direction_angle = single_marker_direction+ math.radians(-90)

                    # Нормализуем угол в диапазон [-π, π]
                    direction_angle = (direction_angle + math.pi) % (2 * math.pi) - math.pi

                    # Обновляем текст направления
                    angle_deg = math.degrees(direction_angle)
                    if -45 <= angle_deg < 45:
                        direction_text = "RIGHT"
                    elif 45 <= angle_deg < 135:
                        direction_text = "DOWN"
                    elif -135 <= angle_deg < -45:
                        direction_text = "UP"
                    else:
                        direction_text = "LEFT"
            elif len(all_centers) == 2:
                # Для двух маркеров используем метод пересечения линий
                avg_center = find_center_from_two_non_opposite_markers(
                    frame_with_markers, markers_centers, direction_angle)
                if avg_center is None:
                    # Если маркеры противоположные, используем среднее или расчетный центр
                    if calculated_center is not None:
                        avg_center = calculated_center
                    else:
                        avg_center = np.mean(all_centers, axis=0).astype(int)
            elif len(all_centers) == 3 and calculated_center is not None:
                # Для трех маркеров с противоположной парой используем расчетный центр
                avg_center = calculated_center
            else:
                # Во всех остальных случаях используем среднее центров
                avg_center = np.mean(all_centers, axis=0).astype(int)

            cv2.circle(frame_with_markers, tuple(avg_center), 8, (255, 0, 0), -1)
            cv2.putText(frame_with_markers, f"Avg: ({avg_center[0]}, {avg_center[1]})",
                        (avg_center[0] + 20, avg_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(frame_with_markers, f"Avg: ({avg_center[0]}, {avg_center[1]})",
                        (10, 80+30+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            #ROBOT INFO
            side_text = "UpSide" if robot_detected_DownSide else "DownSide"
            side_text = "Type position: " + side_text

            cv2.putText(frame_with_markers, side_text,
                        (10, 80+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if direction_angle is not None:
                draw_line_direction(frame_with_markers, tuple(avg_center), direction_angle)
                angle_deg = math.degrees(direction_angle)
                cv2.putText(frame_with_markers, f"Dir: ({round(angle_deg, 3)})",
                            (avg_center[0] + 20, avg_center[1] + 60 - 29),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # ROBOT INFO
                cv2.putText(frame_with_markers, f"Dir: ({round(angle_deg, 3)})",
                            (10, 80+30+30+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame_with_markers, "Dir: Not enough markers",
                            (avg_center[0] + 20, avg_center[1] + 60 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # ROBOT INFO
                cv2.putText(frame_with_markers, "Dir: Not enough markers",
                            (10, 80+30+30+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    fps_history.append(fps)

    # Display current FPS
    cv2.putText(frame_with_markers, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame_with_markers, markers_centers, current_time



def print_fps_stats(fps_history):
    if not fps_history:
        print("No FPS data collected")
        return

    min_fps = min(fps_history)
    avg_fps = statistics.mean(fps_history)
    max_fps = max(fps_history)

    print("\nFPS Statistics:")
    print(f"Minimum FPS: {min_fps:.2f}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Maximum FPS: {max_fps:.2f}")
    print(f"Total frames processed: {len(fps_history)}")


def main():
    # Загружаем калибровочные данные
    calib_data = load_calibration_data()
    if calib_data:
        print("Калибровочные данные успешно загружены")
    else:
        print("Используется изображение без коррекции дисторсии")

    # Optimize OpenCV
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    cv2.ocl.setUseOpenCL(True)

    with open("markers/markers_ids.json", "r") as file:
        markers_ids = json.load(file)
    with open("markers/markers_None.json", "r") as file:
        markers_None = json.load(file)
    with open("markers/markers_off.json", "r") as file:
        markers_dir = json.load(file)

    detector = setup_detection(markers_ids, markers_None)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Устанавливаем разрешение с учетом калибровочных данных
    if calib_data:
        width, height = calib_data["imSize"]
    else:
        width, height = 1920, 1080
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"resolution: {frame_width}x{frame_height}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    video_filename = get_next_video_filename()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    print(f"Press 'q' to exit... Recording video to {video_filename}")

    prev_time = time.time()
    fps_history = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not get frame from camera")
                break

            processed_frame, centers, current_time = process_frame(
                frame, detector, markers_ids, markers_None, markers_dir, prev_time, fps_history, calib_data)
            prev_time = current_time

            out.write(processed_frame)
            cv2.imshow('ArUco Marker Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video saved as {video_filename}")

        # Print FPS statistics
        print_fps_stats(fps_history)

if __name__ == "__main__":
    main()
