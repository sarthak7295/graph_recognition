import PIL.ImageEnhance
import PIL.ImageOps
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract

def merge_lines(lines):
    for line in lines:
        for x1, y1, x2, y2 in line:


    return 0

def is_intersect(rect1, rect2):
    if rect1.min_x > rect2.max_x or rect1.max_x < rect2.min_x:
        return False
    if rect1.min_y > rect2.max_y or rect1.max_y < rect2.min_y:
        return False
    return True