import cv2
import numpy as np
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
img = cv2.VideoCapture('testvid2.mp4')
rert, image = img.read()

# Create a window
cv2.namedWindow('image')

# Initialize HSV min/max values
# Orange PingPong ball
#[5, 75, 165], [30, 245, 255]
hMin = 0
sMin = 0
vMin = 135
hMax = 179
sMax = 60
vMax = 255

# White PingPong ball
#[5, 0, 75], [95, 110, 190]
# hMin = 5
# sMin = 0
# vMin = 75
# hMax = 95
# sMax = 110
# vMax = 190

# Black Pingpong paddle
#[5, 75, 165], [30, 245, 255]
# hMin = 5
# sMin = 75
# vMin = 165
# hMax = 30
# sMax = 245
# vMax = 255



phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# Set minimum and maximum HSV values to display
lower = np.array([hMin, sMin, vMin])
upper = np.array([hMax, sMax, vMax])


while(1):
    
    rert, result = img.read()

    # # Convert to HSV format and color threshold
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, lower, upper)
    # result = cv2.bitwise_and(image, image, mask=mask)

    # Convert to grayscale. 
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) 

    # Blur using 3 * 3 kernel. 
    gray_blurred = cv2.blur(gray, (3, 3)) 

    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray_blurred, 
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                param2 = 30, minRadius = 1, maxRadius = 60) 

    if detected_circles is not None: 

    # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles)) 

        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 

            # Draw the circumference of the circle. 
            cv2.circle(result, (a, b), r, (0, 255, 0), 2) 

            # Draw a small circle (of radius 1) to show the center. qq
            cv2.circle(result, (a, b), 1, (0, 0, 255), 3) 
            cv2.imshow('image', result)
            cv2.waitKey(1) 


    # Display result image
    cv2.imshow('image', result)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


img.release()
cv2.destroyAllWindows()
