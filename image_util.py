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
    i = 150
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


def remove_boxes(img,circles):
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        # cv2.circle(output, (x, y), r,  (0, 255, 0), 4)
        # cv2.circle(img, (x+5, y), r+10, (255, 255, 0), -1)
        cv2.rectangle(img, (x-20, y-20), (x+24, y+20), (0, 0, 0), -1)

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


def detect_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    low_threshold = 100
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    # print(edges)
    # print(img)
    cv2.imshow("skel", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 25  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 15  # minimum number of pixels making up a line
    max_line_gap = 25  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(gray, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    return lines

img = Image.open('myfile1.png').convert('L')
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
# read_circles(b, circles)


############code for line ###########

img1 = cv2.imread('fin.png')
img1 = remove_boxes(img1,circles)
cv2.imwrite("test1.png",img1)
# gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#
# kernel_size = 5
# blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
# low_threshold = 100
# high_threshold = 150
# edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
# rho = 1  # distance resolution in pixels of the Hough grid
# theta = np.pi / 180  # angular resolution in radians of the Hough grid
# threshold = 15  # minimum number of votes (intersections in Hough grid cell)
# min_line_length = 5  # minimum number of pixels making up a line
# max_line_gap = 20  # maximum gap in pixels between connectable line segments
# line_image = np.copy(img1) * 0  # creating a blank to draw lines on
#
# # Run Hough on edge detected image
# # Output "lines" is an array containing endpoints of detected line segments
# lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
skel =cv2.imread("skeleton.png")

# cv2.imshow("skel", skel)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
lines = detect_lines(skel)
line_image = np.copy(img1) * 0  # creating a blank to draw lines on
print(lines.shape)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,255),1)

lines_edges = cv2.addWeighted(img1, 0, line_image, 1, 0)

cv2.imwrite('houghlines3.png',lines_edges)
# ends = cv2.imread('houghlines3.png',0)
#
# corners = cv2.goodFeaturesToTrack(ends, 200, 0.5, 5)
# print(type(corners))
# for corner in corners:
#    for a in corner:
#         print(a)
#         cv2.circle(ends, (tuple(a)), 5, (255,255,0), -1)
#         # cv2.circle(img, (x+5, y), r+10, (255, 255, 0), -1)
# plt.imshow(ends, cmap='gray', interpolation='bicubic')
# plt.show()
# a = pytesseract.image_to_string("fin.png")
# print(a)
# print(a)
# find_circles(img)
