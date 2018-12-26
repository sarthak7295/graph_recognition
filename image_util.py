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
        # cv2.circle(img, (x+5, y), r+10, (255, 255, 0), -1)
        cv2.rectangle(img, (x-15, y-15), (x+24, y+20), (255, 255, 0), -1)
        if i < 99:
            cv2.putText(img, str(i), (x-10, y+10), font, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, str(i), (x - 15, y + 10), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        i = i + 1
    return img


def read_circles(img,circles):
    for (x, y, r) in circles:
        box_img = img[y-15:y+20, x-15:x+24]
        # box_img = cv2.cvtColor(box_img, cv2.COLOR_GRAY2BGR)
        # dst = cv2.fastNlMeansDenoisingColored(box_img, None, 10, 10, 7, 21)
        im_pil = Image.fromarray(box_img)
        converter = PIL.ImageEnhance.Color(im_pil)
        im_pil = converter.enhance(50)
        converter2 = PIL.ImageEnhance.Contrast(im_pil)
        im_pil = converter2.enhance(50)
        open_cv_image = np.array(im_pil)
        # plt.imshow(open_cv_image, cmap='gray', interpolation='bicubic')
        # plt.show()
        a = pytesseract.image_to_string(open_cv_image, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        print(a)

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
cv2.imwrite('fin.png',a)
b = cv2.imread('fin.png',0)
read_circles(b, circles)
# a = pytesseract.image_to_string("fin.png")
# print(a)
# print(a)
# find_circles(img)
