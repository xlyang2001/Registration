import os

import cv2
import numpy as np
import torch

import time

from model import Reg

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = r'Model/320_RegModel.pth'

model = Reg()
model.load_state_dict(torch.load(model_path))
model.to(device)


model.eval()

xray = cv2.imread("Data/DRR_9000.png", cv2.IMREAD_GRAYSCALE)
xray = cv2.resize(xray, (128, 128))
xray = torch.tensor(xray, dtype=torch.float32)
xray = xray.reshape((1, 1, 128, 128))
xray = xray.to(device)

with torch.no_grad():
    _, _, _, result = model(xray)
    result = result.reshape((1, 6))

print(result)




