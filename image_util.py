import PIL.ImageEnhance
import PIL.ImageOps
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
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
    c2 = 11
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, image_cols / 10, param1=c1, param2=c2,minRadius=0,maxRadius=15)
    # print(circles.shape)
    # print(circles)
    # plt.imshow(img, cmap='gray', interpolation='bicubic')
    # plt.show()
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        return circles
    return 0


def write_numbers(img,circles):
    i = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        # cv2.circle(output, (x, y), r,  (0, 255, 0), 4)
        i = i + 1
        cv2.circle(img, (x+5, y), r+10, (255, 255, 0), -1)
        if i < 99:
            cv2.putText(img, str(i), (x-10, y+10), font, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, str(i), (x - 10, y + 10), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return img


img = Image.open('myfile.png').convert('L')
img = covert_image_greyscale(img)
img.save('grey.png')
img = cv2.imread('grey.png',0)
cv2.imwrite('grey3.png',img)
# img = cv2.medianBlur(img,5)
# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.show()
circles = find_circles(cv2.bitwise_not(img))
img = cv2.imread('grey3.png',0)
a = write_numbers(img,circles)
plt.imshow(a, cmap='gray', interpolation='bicubic')
plt.show()
a = pytesseract.image_to_string("test.png")
print(a)
# print(a)
# find_circles(img)
