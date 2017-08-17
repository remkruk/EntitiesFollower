import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    image, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # screenCnt = None
    # for c in cnts:
    #     # approximate the contour
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    #
    #     # if our approximated contour has four points, then
    #     # we can assume that we have found our screen
    #     if len(approx) == 4:
    #         screenCnt = approx
    #         break

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

    # screenCnt = None
    # if len(cnts) > 0 :
    #     screenCnt = max(cnts, key = cv2.contourArea)
    #
    # cv2.drawContours(frame, cnts, -1, (0, 255, 0), 3)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    # cv2.imshow('contours', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()