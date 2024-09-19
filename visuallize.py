import torch
import numpy as np
import matplotlib.pyplot as plt

# 模拟你给的检测结果 (假设在 CPU 上)
detections = torch.tensor([[-4.3695e+00, -2.8651e+00,  1.8326e+01,  ...,  1.6935e-05,
          1.5665e-01,  1.7000e+01],
        [ 4.5222e-01, -4.1146e+00,  3.1893e+01,  ...,  2.6125e-06,
          5.8369e-02,  3.4000e+01],
        [ 4.3658e+00, -4.6296e+00,  3.9161e+01,  ...,  5.5332e-08,
          4.6939e-02,  3.5000e+01],
        ...,
        [ 1.2648e+03,  1.2659e+03,  1.2955e+03,  ...,  4.5900e-05,
          2.1923e-03,  1.2000e+01],
        [ 1.2727e+03,  1.2631e+03,  1.3680e+03,  ...,  1.9642e-05,
          5.1685e-03,  1.2000e+01],
        [ 1.3454e+03,  1.3535e+03,  1.3863e+03,  ...,  4.3670e-03,
          2.1294e-03,  1.2000e+01]], device='cuda:0')

import cv2
image_path = '/workspace/assets/00000.png'
image = cv2.imread(image_path)
for detection in detections:
    x_min, y_min, x_max, y_max = map(int, detection[:4])
    confidence = detection[4].item()  # 置信度
    class_confidence = detection[5].item()  
    class_id = int(detection[6].item()) 
    color = (0, 255, 0) 
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)
cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite("/workspace/assets/00000.jpg", image)


