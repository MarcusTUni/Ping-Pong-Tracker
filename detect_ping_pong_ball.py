#Detect green

#Testing Github

#Changing the game

import cv2
import numpy as np

def nothing(x):
    pass

# Load image
img = cv2.VideoCapture(0) # 0 refers to camera, eg number
# image = cv2.GaussianBlur(image, (7, 7), 0) # for blurring the image, can use if want to

# Create a window
cv2.namedWindow('image')

#[5, 75, 165], [30, 245, 255]
# Initialize HSV min/max values
hMin = 5
sMin = 75
vMin = 165
hMax = 30
sMax = 245
vMax = 255
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while(1):
    
    rert, image = img.read()

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Display result image
    cv2.imshow('image', result)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
