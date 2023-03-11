import math

import cv2
import numpy as np
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
asl_classifier = Classifier("ASL_model.h5")
bsl_classifier = Classifier("BSL_model.h5")
offset = 20
imgSize = 600
folder = "Data/BSL/A"
# File That use to save the images
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
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
            prediction, index = asl_classifier.getPrediction(img)
            prediction, index = bsl_classifier.getPrediction(img)
            predicted_label = labels[index]
            cv2.putText(imgWhite, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape

            hGap = math.ceil((imgSize - hCal) / 2)

            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)



   # Every when hand sign learn it has to be act like a keyboard "A" in the data prediction in the gui create label or text field - pamudu
    # create button called "Next" which will keep going the word. - pamudu
    # create button called "clear"
    # Every time when it learn the hand sign that letter has to be printed in the textfield or label.
    # when we press a button in the or press the button in key board have to trigger the text field word should be read by ai. - create autometic generated text in the file and make it read by ai. (vinuka)
    # text filed in upper part which will give suggestion word for to communicate - Sakeen
    # In the gui there has to a button called "Space" only this feature should trigger whenever we space bar also.
    # merge code.
