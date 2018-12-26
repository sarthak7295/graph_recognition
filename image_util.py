import PIL.ImageEnhance
import PIL.ImageOps
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt


def covert_image_greyscale(img):
    img = img.resize((500, 500), Image.ANTIALIAS)
    img = PIL.ImageOps.invert(img)
    converter = PIL.ImageEnhance.Color(img)
    img = converter.enhance(50)
    converter2 = PIL.ImageEnhance.Contrast(img)
    img = converter2.enhance(50)
    return img


def find_circles(img):
    image_cols = 500
    dp = 1
    c1 = 100
    c2 = 15
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, image_cols / 10, param1=c1, param2=c2,minRadius=0,maxRadius=50)
    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,1,1)
    print(circles)
    output = img.copy()
    # plt.imshow(img, cmap='gray', interpolation='bicubic')
    # plt.show()
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # show the output image
        cv2.imshow("output", np.hstack([img, output]))
        cv2.waitKey(0)
    return 1


img = Image.open('myfile.png').convert('L')
img = covert_image_greyscale(img)
img.save('grey.png')
img = cv2.imread('grey.png',0)
# img = cv2.medianBlur(img,5)
# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.show()
find_circles(cv2.bitwise_not(img))
