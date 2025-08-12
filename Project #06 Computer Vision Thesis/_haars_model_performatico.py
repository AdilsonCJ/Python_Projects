import cv2
import numpy as np
import pandas as pd

# Caminhos dos arquivos
video_path = './videos/sample5_highdef.mp4'
mask_path = './masks/sample5_mask.png'
haar_models = [
    {
        "name": "sample5_modelo1haars_sensivel",
        "haar_path": "./haarscascade_models/haarcascade_upperbody.xml",
        "scaleFactor": 1.01,
        "minNeighbors": 3,
        "minSize": (20, 20)
    },
    {
        "name": "sample5_modelo2haars_conservador",
        "haar_path": "./haarscascade_models/haarcascade_upperbody.xml",
        "scaleFactor": 1.05,
        "minNeighbors": 8,
        "minSize": (40, 40)
    },
    {
        "name": "sample5_modelo3haars_fullbody",
        "haar_path": "./haarscascade_models/haarcascade_upperbody.xml",
        "scaleFactor": 1.03,
        "minNeighbors": 5,
        "minSize": (30, 30)
    }
]

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

def processar_video(haar_path, scaleFactor, minNeighbors, minSize, nome_saida):
    
    cap = cv2.VideoCapture(video_path)
    detector = cv2.CascadeClassifier(haar_path)
    frame_count = 0
    frame_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_resized = cv2.resize(frame, (mask.shape[1], mask.shape[0]))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        masked_gray = cv2.bitwise_and(gray, mask)

        bodies = detector.detectMultiScale(masked_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
        count = len(bodies)

        print(f"Frame {frame_count}: {count} pessoa(s) detectada(s)")
        frame_data.append({"frame": frame_count, "pessoas_detectadas": count})

        # for (x, y, w, h) in bodies:
        #     cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow(f'Detecção - {nome_saida}', frame_resized)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    # cv2.destroyAllWindows()

    df = pd.DataFrame(frame_data)
    arquivo_saida = f"resultados_contagem_haar_{nome_saida}.csv"
    df.to_csv(arquivo_saida, index=False)

# Loop sobre os cenários
for modelo in haar_models:
    processar_video(
        haar_path=modelo["haar_path"],
        scaleFactor=modelo["scaleFactor"],
        minNeighbors=modelo["minNeighbors"],
        minSize=modelo["minSize"],
        nome_saida=modelo["name"]
    )