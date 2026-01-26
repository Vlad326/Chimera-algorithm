
def GetResultsFrame(model, frame, conf, imgsz=(736, 1280)):
    results = model(frame, conf=conf, verbose=False, imgsz=imgsz, max_det=1)
    try:
        obb = results[0].obb.xyxyxyxy.cpu().numpy().reshape(4, 2)
    except ValueError:
        obb = None


    return {"results":results, 'obb':obb}
