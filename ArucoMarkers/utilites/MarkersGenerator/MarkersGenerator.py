import cv2
import os

# Define the dictionary we want to use
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


for marker_id in range(5, 9):
    marker_size = 250  # Size in pixels
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    base_name = f'markers/marker_id{marker_id}'
    extension = '.png'
    counter = 1
    output_filename = base_name + extension

    while os.path.exists(output_filename):
        output_filename = f"{base_name}_{counter}{extension}"
        counter += 1


    cv2.imwrite(f'{output_filename}', marker_image)
