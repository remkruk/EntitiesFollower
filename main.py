import numpy as np
import cv2
import argparse

def search_shape(contours, frame, vertices):
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon=0.01 * peri, closed=True)

        if len(approx) == vertices:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)


parser = argparse.ArgumentParser("Find specific figures")
parser.add_argument('vertices', metavar='v', type= int, help= 'specify vertices amount of figure you want to search for')

args = vars(parser.parse_args())

vertices = args["vertices"]
print(vertices)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    image, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    search_shape(cnts, frame, vertices)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



