import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # hsv hue sat value
    lower_pink = np.array((140,50,30))
    upper_pink = np.array((180,255,255))

    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    kernel = np.ones((5,5), np.uint8)

    erosion = cv2.erode(mask, kernel, iterations = 1)

    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel) # removes background noise

    res = cv2.bitwise_and(frame, frame, mask=opening)

    median = cv2.medianBlur(res, 15)

    cv2.imshow('frame', frame)
    cv2.imshow('median', median)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()