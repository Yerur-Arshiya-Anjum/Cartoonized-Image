import numpy as np
import cv2

NumOfDown = 2        # number of down sampling steps
NumOfBilateral = 7   # number of bilateral filtering steps

img = cv2.imread("images/car2.jpg")

# Resizing
img = cv2.resize(img,(600,600))

# down sampling image using gaussian pyramid
img_color = img
for _ in range(NumOfDown):
    img_color = cv2.pyrDown(img)

# repeatedly apply small bilateral filter instead of applying one large filter
for _ in range(NumOfBilateral):
    img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

# Un sample image to original size
for i in range(NumOfDown):
    img_color = cv2.pyrUp(img_color)

img_gray = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
img_blur = cv2.medianBlur(img_gray, 7)

img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)

# Convert back to color, bit_AND with color image
img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
img_cartoon=cv2.bitwise_and(img, img_edge)

# Display
stack = np.hstack([img, img_cartoon])
cv2.imshow("original-cartooned", stack)
cv2.waitKey(0)

