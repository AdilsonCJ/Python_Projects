import cv2
import numpy as np
import torch
import pandas as pd
import torchvision.transforms as transforms
from collections import OrderedDict
import math
from csrnet_model import CSRNet
from scipy.ndimage import gaussian_filter

# ----- Configuração do dispositivo e modelo -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSRNet().to(device)
checkpoint = torch.load('./vgg16_models/PartAmodel_best.pth', map_location=device, weights_only=False)
state_dict = checkpoint.get('state_dict', checkpoint)
new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
model.load_state_dict(new_state_dict)
model.eval()

# ----- Máscara -----
mask = cv2.imread('./masks/sample5_mask.png', cv2.IMREAD_GRAYSCALE)

_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#mask = cv2.resize(mask, (1024, 768)) / 255.0

# ----- Função de processamento do vídeo -----
def process_video(resize_shape, use_gaussian, sigma_value, normalize_params, output_csv):
    
    cap = cv2.VideoCapture('./videos/sample5_highdef.mp4')
    frame_count = 0
    results = []

    # Define a transformação (sem pré-processamento visual)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_params['mean'], std=normalize_params['std'])
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (resize_shape[1], resize_shape[0]))  # largura x altura

        resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        resized_mask = resized_mask.astype(np.uint8)  # Garante tipo correto
        masked_frame = cv2.bitwise_and(frame, frame, mask=resized_mask)

        input_tensor = transform(masked_frame).unsqueeze(0).to(device)

        with torch.no_grad():
            density_map = model(input_tensor).cpu().squeeze().numpy()
            if use_gaussian:
                density_map = gaussian_filter(density_map, sigma=sigma_value)
            count = density_map.sum()

        results.append({'frame': frame_count, 'pessoas_detectadas': round(count)})

    cap.release()
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

# ----- Três cenários -----
cenarios = [
    {
        "resize_shape": (768, 1024),  # padrão
       "use_gaussian": False,
        "sigma_value": 0,
        "normalize_params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "output_csv": "sample5_cenario_resize_original.csv"
    },
    {
        "resize_shape": (640, 960),  # resolução reduzida
        "use_gaussian": False,
        "sigma_value": 0,
        "normalize_params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "output_csv": "sample5_cenario_resize_640x960.csv"
    },
    {
        "resize_shape": (768, 1024),  # mantém original
        "use_gaussian": True,         # aplica filtro gaussiano
        "sigma_value": 1.0,
        "normalize_params": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},  # normalização alternativa
        "output_csv": "sample5_cenario_gaussian_normalization_custom.csv"
    }
]

# ----- Executar os testes -----
for c in cenarios:
    process_video(**c)
