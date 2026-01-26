
def GetResultsFrame(model, frame, conf, imgsz=(736, 1280)):
    results = model(frame, conf=conf, verbose=False, imgsz=imgsz, max_det=1)

    boxes = results[0].boxes.xyxy.cpu().numpy()

    centers = []
    for box in boxes:
        x1, y1, x2, y2 = box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        centers.append([center_x, center_y])

    return {"boxes":boxes, "centers":centers, "resultsModel":results}
