import cv2
import numpy as np
import pandas as pd

def apply_mask(frame, mask_img):
    mask = cv2.imread(mask_img, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    return mask

def process_video(video_path, mask_img, params, nome_saida):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cap = cv2.VideoCapture(video_path)
    ret, sample_frame = cap.read()
    if not ret:
        print("Erro ao carregar vídeo")
        return

    mask = apply_mask(sample_frame, mask_img)
    frame_data = []
    frame_number = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3,3), 0)

        # Detecta pessoas com HOG e os parâmetros do cenário
        boxes, weights = hog.detectMultiScale(
            gray,
            winStride=params['winStride'],
            padding=params['padding'],
            scale=params['scale'],
            hitThreshold=params['hitThreshold']
        )

        count_people = 0
        for i, (x, y, w, h) in enumerate(boxes):
            if weights[i] < 0.5:
                continue
            cx, cy = x + w//2, y + h//2
            if mask_resized[cy, cx] > 0:
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                count_people += 1

        print(f"Frame {frame_number}: Pessoas detectadas = {count_people}")
        frame_data.append({"frame": frame_number, "pessoas_detectadas": count_people})

    cap.release()
    # cv2.destroyAllWindows()

    df = pd.DataFrame(frame_data)
    arquivo_saida = f"resultado_contagem_hog_{nome_saida}.csv"
    df.to_csv(arquivo_saida, index=False)

if __name__ == "__main__":
    video_path = './videos/sample5_highdef.mp4'
    mask_img = './masks/sample5_mask.png'
     

    # Definição dos três cenários com diferentes parâmetros
    cenarios = [
        {
            "name": "sample5_modelo1hog_sensivel",
            "params": {
                "winStride": (4,4),
                "padding": (8,8),
                "scale": 1.01,
                "hitThreshold": 0.0
            }
        },
        {
            "name": "sample5_modelo2hog_conservador",
            "params": {
                "winStride": (8,8),
                "padding": (16,16),
                "scale": 1.05,
                "hitThreshold": 0.5
            }
        },
        {
            "name": "sample5_modelo3hog_intermediario",
            "params": {
                "winStride": (6,6),
                "padding": (12,12),
                "scale": 1.03,
                "hitThreshold": 0.2
            }
        }
    ]

    # Loop sobre os cenários
    for cenario in cenarios:
        process_video(video_path, mask_img, cenario["params"], cenario["name"])