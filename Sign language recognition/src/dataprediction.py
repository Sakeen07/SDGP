import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier_asl = Classifier("ASL_model.h5", "Model.txt")
classifier_bsl = Classifier("BSL_model.h5", "Model.txt")
offset = 20
imgSize = 600
prediction_list = []
label_count = {}
final_label = ""
prev_label = ""
start_time = time.time()

@app.route('/post-request', methods=['POST'])
def receive_data():
    data = request.get_json()
    # Process the data here
    response_data = {'message': 'Data received'}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run()

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

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape

            hGap = math.ceil((imgSize - hCal) / 2)

            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Use both classifiers to predict the sign language gesture
        prediction_asl = classifier_asl.getPrediction(imgWhite)[0]
        prediction_bsl = classifier_bsl.getPrediction(imgWhite)[0]

        # Choose the prediction with higher probability
        if prediction_asl[1] > prediction_bsl[1]:
            final_prediction = prediction_asl[0]
        else:
            final_prediction = prediction_bsl[0]

        # Add the prediction to the list
        prediction_list.append(final_prediction)

        # Count the labels in the prediction list
        label_count = {}
        for label in prediction_list:
            if label not in label_count:
                label_count[label] = 1
            else:
                label_count[label] += 1

        # Check if a label has occurred for at least three seconds
        if time.time() - start_time >= 3:
            max_count = 0
            for label, count in label_count.items():
                if count > max_count:
                    max_count = count
                    final_label = label
            if prev_label != final_label:
                start_time = time.time()
            prev_label = final_label

        # Display the predicted label at the bottom of the frame
        cv2.rectangle(img, (0, img.shape[0] - 50), (img.shape[1], img.shape[0]), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, str(final_label), (int(img.shape[1] / 2) - 50, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

# Show the result of the ingWhite to the main frame -
# change the dataset for the ASL model and create new dataset file - done (added new code like time)
# combine both results and get the average from both to get the final prediction - done.

# Combine the letter with the different prediction to create a word
# Connect the text word to the AI generated voice
# Connect the voice and text with the translation

# Connect with the frontend.
