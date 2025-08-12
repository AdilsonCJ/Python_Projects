import cv2
import numpy as np

video_path = './videos/sample1_highdef_trim.mp4'
mask_path = './masks/sample1New_mask.png'

cap = cv2.VideoCapture(video_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Erro ao ler o primeiro frame.")
    exit()

frame_h, frame_w = frame.shape[:2]
mask = cv2.resize(mask, (frame_w, frame_h))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_count = 0

cv2.namedWindow('Frame com Máscara', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame com Máscara', 800, 600)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(masked_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    resized_frame = cv2.resize(masked_frame, (600, 600))

    cv2.imshow('Frame com Máscara', resized_frame)

    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    elif key == ord(' '):
        continue

cap.release()

cv2.destroyAllWindows()