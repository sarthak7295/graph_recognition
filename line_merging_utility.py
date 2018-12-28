import PIL.ImageEnhance
import PIL.ImageOps
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract


class Rect:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2


def merge_lines(lines):
    for line in lines:
        for x11, y11, x12, y12 in line:
            for line in lines:
                for x21, y21, x22, y22 in line:
                    # if ( x11, y11, x12, y12 != x21, y21, x22, y22 ):
                        rect1 = Rect(x11,y11,x12,y12)
                        rect2 = Rect(x21, y21, x22, y22)
                        intersecting_rectangle = is_intersect_two(rect1, rect2)

    return 0


def is_intersect(rect1, rect2):
    if rect1.min_x > rect2.max_x or rect1.max_x < rect2.min_x:
        return False
    if rect1.min_y > rect2.max_y or rect1.max_y < rect2.min_y:
        return False
    return True


def is_intersect_two(rect1, rect2):
    if min(rect1.x1, rect1.x2) > max(rect2.x1, rect2.x2) or max(rect1.x1, rect1.x2) < min(rect2.x1, rect2.x2):
        return False
    if min(rect1.y1, rect1.y2) > max(rect2.y1, rect2.y2) or max(rect1.y1, rect1.y2) < min(rect2.y1, rect2.y2):
        return False
    return True