import cv2
import numpy as np
import os


def nothing(x):
    pass


image_x, image_y = 64, 64

from keras.models import load_model

classifier = load_model('Model.h5')

# original = cv2.imread("5.png")
fileEntry = []
for file in os.listdir("Data/Final Data"):
    if file.endswith(".png"):
        fileEntry.append(file)


def imgprocessing():
    image_to_compare = cv2.imread("./Data/Final Data/A/0.jpg")
    original = cv2.imread("1.png")
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    ratio = 0.6
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_points.append(m)
    # print(len(good_points))
    fin = len(kp_1) - len(kp_2)
    print(abs(fin))
    if abs(fin) <= 2:
        return 'space'

    result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)

    # cv2.imshow("result", result)
    # cv2.imshow("Original", original)
    # cv2.imshow("Duplicate", image_to_compare)
    # cv2.destroyAllWindows()


def predictor():
    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img('1.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    for i in range(len(fileEntry)):
        image_to_compare = cv2.imread("./SampleGestures/" + fileEntry[i])
        original = cv2.imread("1.png")
        sift = cv2.xfeatures2d.SIFT_create()
        kp_1, desc_1 = sift.detectAndCompute(original, None)
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)

        good_points = []
        ratio = 0.6
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_points.append(m)
        print(fileEntry[i])
        if (abs(len(good_points) + len(matches)) > 20):
            gesname = fileEntry[i]
            gesname = gesname.replace('.png', '')
            return gesname
    # print(good_points)
    # if(abs(fin)<=2):
    # return 'space'
    if result[0][0] == 1:
        return 'A'
    elif result[0][1] == 1:
        return 'B'
    elif result[0][2] == 1:
        return 'C'
    elif result[0][3] == 1:
        return 'D'
    elif result[0][4] == 1:
        return 'E'
    elif result[0][5] == 1:
        return 'F'
    elif result[0][6] == 1:
        return 'G'
    elif result[0][7] == 1:
        return 'H'
    elif result[0][8] == 1:
        return 'I'
    elif result[0][9] == 1:
        return 'J'
    elif result[0][10] == 1:
        return 'K'
    elif result[0][11] == 1:
        return 'L'
    elif result[0][12] == 1:
        return 'M'
    elif result[0][13] == 1:
        return 'N'
    elif result[0][14] == 1:
        return 'O'
    elif result[0][15] == 1:
        return 'P'
    elif result[0][16] == 1:
        return 'Q'
    elif result[0][17] == 1:
        return 'R'
    elif result[0][18] == 1:
        return 'S'
    elif result[0][19] == 1:
        return 'T'
    elif result[0][20] == 1:
        return 'U'
    elif result[0][21] == 1:
        return 'V'
    elif result[0][22] == 1:
        return 'W'
    elif result[0][23] == 1:
        return 'X'
    elif result[0][24] == 1:
        return 'Y'
    elif result[0][25] == 1:
        return 'Z'


cam = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cv2.namedWindow("Sign language Recognition")

img_text = ''
img_text1 = ''
# z5=1
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.putText(frame, img_text1, (80, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("ASL Recognition", frame)
    cv2.imshow("mask", mask)

    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    img_text = predictor()

    if cv2.waitKey(1) == 27:
        break

# img_text1 = imgprocessing()


cam.release()
cv2.destroyAllWindows()
# Combine the letter with the different prediction to create a word
# Connect the text word to the AI generated voice
# Connect the voice and text with the translation

# Connect with the frontend.

# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import time
# from flask import Flask, request, jsonify
#
# app = Flask(_name_)
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifier_asl = Classifier("ASL_model.h5", "Model.txt")
# classifier_bsl = Classifier("BSL_model.h5", "Model.txt")
# offset = 20
# imgSize = 600
# prediction_list = []
# label_count = {}
# final_label = ""
# prev_label = ""
# start_time = time.time()
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     image_base64 = data['image']
#     img = cv2.imdecode(np.fromstring(base64.b64decode(
#         image_base64), np.uint8), cv2.IMREAD_COLOR)
#     hands, img = detector.findHands(img)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#
#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#         imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
#         imgCropShape = imgCrop.shape
#         aspectRatio = h / w
#
#         if aspectRatio > 1:
#             k = imgSize / h
#             wCal = math.ceil(k * w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             imgResizeShape = imgResize.shape
#             wGap = math.ceil((imgSize - wCal) / 2)
#             imgWhite[:, wGap:wCal + wGap] = imgResize
#
#         else:
#             k = imgSize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             imgResizeShape = imgResize.shape
#
#             hGap = math.ceil((imgSize - hCal) / 2)
#
#             imgWhite[hGap:hCal + hGap, :] = imgResize
#
#         # Use both classifiers to predict the sign language gesture
#         prediction_asl = classifier_asl.getPrediction(imgWhite)[0]
#         prediction_bsl = classifier_bsl.getPrediction(imgWhite)[0]
#
#         # Choose the prediction with higher probability
#         if prediction_asl[1] > prediction_bsl[1]:
#             final_prediction = prediction_asl[0]
#         else:
#             final_prediction = prediction_bsl[0]
#
#         # Add the prediction to the list
#         prediction_list.append(final_prediction)
#
#         # Count the labels in the prediction list
#         label_count = {}
#         for label in prediction_list:
#             if label not in label_count:
#                 label_count[label] = 1
#             else:
#                 label_count[label] += 1
#
#         # Check if a label has occurred for at least three seconds
#         if time.time() - start_time >= 3:
#             max_count = 0
#             for label, count in label_count.items():
#                 if count > max_count:
#                     max_count = count
#                     final_label = label
#             if prev_label != final_label:
#                 start_time = time.time()
#             prev_label = final_label
#
#         print(final_label)
#
#         # Display the predicted label at the bottom of the frame
#         cv2.rectangle(
#             img, (0, img.shape[0] - 50), (img.shape[1], img.shape[0]), (255, 255, 255), cv2.FILLED)
