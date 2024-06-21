import os
from cvzone.HandTrackingModule import HandDetector
import cv2

path = "D:\\OPEN CV\\Resources"
pathimages = sorted(os.listdir(path), key=len)
cap = cv2.VideoCapture(0)
imgnumber = 0
annotations = []
hs, ws = int(120 * 1.5), int(215 * 1.5)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(3, 1200)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# List of colors
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
color_index = 0
current_color = colors[color_index]

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=True)

    if imgnumber < len(pathimages):
        folderimage = os.path.join(path, pathimages[imgnumber])
        slide = cv2.imread(folderimage)
        if slide is None:
            print(f"Error: Could not read image {folderimage}.")
            continue

        height, width, _ = img.shape
        center_y = height // 2
        cv2.line(img, (0, center_y), (width, center_y), (0, 255, 0), 10)

        h, w, _ = slide.shape
        imgsmall = cv2.resize(img, (ws, hs))
        slide[0:hs, w - ws:w] = imgsmall
        slide = cv2.resize(slide, (1080, 720))

        if hands:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            lmlist = hand['lmList']
            indexfinger = lmlist[8][0], lmlist[8][1]

            if fingers == [0, 0, 0, 0, 1]:  # Next slide
                if imgnumber < len(pathimages) - 1:
                    imgnumber += 1
                    annotations = []  # Clear annotations for new slide
                else:
                    print("LIMIT EXCEEDED")

            if fingers == [1, 0, 0, 0, 0]:  # Previous slide
                if imgnumber > 0:
                    imgnumber -= 1
                    annotations = []  # Clear annotations for previous slide
                else:
                    print("OUT OF BOUNDS")

            if fingers == [0, 1, 1, 0, 0]:  # Draw
                cv2.circle(slide, indexfinger, 10, current_color, cv2.FILLED)
                annotations.append((indexfinger, current_color))

            if fingers == [0, 1, 1, 1, 0] and annotations:  # Erase last annotation
                annotations.pop(-1)

            if fingers == [0, 1, 1, 1, 1]:  # Change color
                color_index = (color_index + 1) % len(colors)
                current_color = colors[color_index]
                print(f"Changed color to {current_color}")

        # Draw all annotations
        for i in range(1, len(annotations)):
            if annotations[i - 1][1] == annotations[i][1]:  # Check if colors match
                cv2.line(slide, annotations[i - 1][0], annotations[i][0], annotations[i][1], 10)
            else:
                cv2.circle(slide, annotations[i][0], 10, annotations[i][1], cv2.FILLED)

        cv2.imshow("Slide", slide)

    else:
        print(f"Error: No more images to display.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
