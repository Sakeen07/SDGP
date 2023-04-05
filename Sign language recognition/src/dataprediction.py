from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from fastapi.middleware.cors import CORSMiddleware
import cv2
import math
import numpy as np
import threading

app = FastAPI()
lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 600
classifier = Classifier("Model.h5", "Model.txt")
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z"]
global label
label = ""

def detect_objects(frame):
    if frame is None or frame.size == 0:
        return {"Label": "", "Result": None}
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_output = img.copy()
    hands, img = detector.findHands(frame)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        cv2.putText(img_output, labels[index], (x, y - 20),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(img_output, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)
        return {"Label": labels[index], "Result": img_output}
    else:
        return {"Label": "", "Result": img_output}



