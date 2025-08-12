from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np
import pandas as pd

def run_yolo_pipeline(conf_threshold, iou_threshold_yolo, tracker_params, output_csv):

    cap = cv2.VideoCapture('./videos/sample5_highdef.mp4')
    yolo_mask = cv2.imread('./masks/sample5_mask.png')
    model = YOLO('./yolo_models/yolov8m.pt')

    tracker = Sort(max_age=tracker_params['max_age'],
                   min_hits=tracker_params['min_hits'],
                   iou_threshold=tracker_params['iou_threshold'])

    frame_count = 0
    dados_frame = []

    while True:
        success, img = cap.read()
        if not success:
            break
        frame_count += 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # comentado para manter RGB

        mask_resized = cv2.resize(yolo_mask, (img.shape[1], img.shape[0]))
        imgRegion = cv2.bitwise_and(img, mask_resized)

        results = model(imgRegion, stream=True, conf=conf_threshold, iou=iou_threshold_yolo)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if cls == 0 and conf >= conf_threshold:
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)
        pessoas_detectadas = len(resultsTracker)

        dados_frame.append({
            "frame": frame_count,
            "pessoas_detectadas": pessoas_detectadas
        })

        for result in resultsTracker:
            x1, y1, x2, y2, id = map(int, result)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2)
            cvzone.putTextRect(img=img, text=f'{id}', pos=(max(0, x1), max(35, y1)),
                               scale=1, thickness=2, offset=3)

        cv2.imshow("YOLO Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(dados_frame)
    df.to_csv(output_csv, index=False)

# ----- Definição dos cenários -----
cenarios = [
    {
        "conf_threshold": 0.3,
        "iou_threshold_yolo": 0.3,
        "tracker_params": {"max_age": 10, "min_hits": 2, "iou_threshold": 0.1},
        "output_csv": "yolo_sample5_cenario_baseline.csv"
    },
    {
        "conf_threshold": 0.5,
        "iou_threshold_yolo": 0.3,
        "tracker_params": {"max_age": 10, "min_hits": 2, "iou_threshold": 0.1},
        "output_csv": "yolo_sample5_cenario_conf05.csv"
    },
    {
        "conf_threshold": 0.3,
        "iou_threshold_yolo": 0.5,
        "tracker_params": {"max_age": 5, "min_hits": 3, "iou_threshold": 0.2},
        "output_csv": "yolo_sample5_cenario_tracker_tuned.csv"
    }
]

# ----- Executa os cenários -----
for c in cenarios:
    run_yolo_pipeline(**c)