import PIL.ImageEnhance
import PIL.ImageOps
from PIL import Image
import cv2
import numpy as np
import skeleton_creater
import line_merging_utility as lmu

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
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        return circles
    return 0


def write_numbers(img,circles):
    i = 150
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x, y, r) in circles:
        cv2.rectangle(img, (x-15, y-15), (x+24, y+20), (255, 255, 0), -1)
        if i < 99:
            cv2.putText(img, str(i), (x-10, y+10), font, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, str(i), (x - 15, y + 10), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        i = i + 1
    return img


def remove_boxes(img,circles):
    for (x, y, r) in circles:
        cv2.rectangle(img, (x-20, y-20), (x+24, y+20), (0, 0, 0), -1)
    return img


def detect_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 25  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 15  # minimum number of pixels making up a line
    max_line_gap = 25  # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(gray, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    return lines


img = Image.open('myfile1.png').convert('L')
img = covert_image_greyscale(img)
img = np.array(img)
circles = find_circles(cv2.bitwise_not(img))
img = write_numbers(img,circles)

############code for line ###########

img = remove_boxes(img,circles)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = skeleton_creater.create_skeleton(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
lines = detect_lines(img)
line_image = np.copy(img) * 0  # creating a blank to draw lines on
# print(lmu.merge_lines(lines))
print(lines.shape)
lines = lmu.merge_lines(lines)
lines = list(set(lines))
lines = np.array(lines)
print(lines.shape)
# print(lines)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 255), 1)

lines_edges = cv2.addWeighted(img, 0, line_image, 1, 0)
# cv2.rectangle(lines_edges, (100, 100), (0, 0), (0, 255, 0), -1)
cv2.imwrite('houghlines3.png',lines_edges)
