import math
import numpy as np

class Rect:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2


def merge_lines(lines):
    a = []
    print(type(a))
    for line in lines:
        for x11, y11, x12, y12 in line:
            for line in lines:
                for x21, y21, x22, y22 in line:
                    # if ( x11, y11, x12, y12 != x21, y21, x22, y22 ):
                        rect1 = Rect(x11,y11,x12,y12)
                        rect2 = Rect(x21, y21, x22, y22)
                        intersecting_rectangle = is_intersect_two(rect1, rect2)
                        good_angle = check_angle(rect1,rect2)
                        if intersecting_rectangle and good_angle :
                            temp = x1, y1, x2, y2 = get_max_distance_pair(rect1,rect2)
                            a.append(list(temp))
    return a


def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude


def get_max_distance_pair(rect1,rect2):
    fin_x1,fin_y1,fin_x2,fin_y2 = rect1.x1,rect1.y1,rect1.x2,rect1.y2
    x1, y1, x2, y2 = rect1.x1, rect1.y1, rect1.x2, rect1.y2
    fin_dist = lineMagnitude(x1,y1,x2,y2)

    x1, y1, x2, y2 = rect1.x1, rect1.y1, rect2.x1, rect2.y1
    dist = lineMagnitude(x1,y1,x2,y2)
    if(dist >= fin_dist):
        fin_x1, fin_y1, fin_x2, fin_y2 = x1, y1, x2, y2
        fin_dist = dist

    x1, y1, x2, y2 = rect1.x1, rect1.y1, rect2.x2, rect2.y2
    dist = lineMagnitude(x1, y1, x2, y2)
    if (dist >= fin_dist):
        fin_x1, fin_y1, fin_x2, fin_y2 = x1, y1, x2, y2
        fin_dist = dist

    x1, y1, x2, y2 = rect1.x2, rect1.y2, rect2.x1, rect2.y1
    dist = lineMagnitude(x1, y1, x2, y2)
    if (dist >= fin_dist):
        fin_x1, fin_y1, fin_x2, fin_y2 = x1, y1, x2, y2
        fin_dist = dist

    x1, y1, x2, y2 = rect1.x2, rect1.y2, rect2.x2, rect2.y2
    dist = lineMagnitude(x1, y1, x2, y2)
    if (dist >= fin_dist):
        fin_x1, fin_y1, fin_x2, fin_y2 = x1, y1, x2, y2
        fin_dist = dist

    x1, y1, x2, y2 = rect2.x1, rect2.y1, rect2.x2, rect2.y2
    dist = lineMagnitude(x1, y1, x2, y2)
    if (dist >= fin_dist):
        fin_x1, fin_y1, fin_x2, fin_y2 = x1, y1, x2, y2

    return fin_x1,fin_y1,fin_x2,fin_y2

def check_angle(rect1,rect2):
    vector_one = np.array([rect1.x2 - rect1.x1, rect1.y2 - rect1.y1])
    vector_two = np.array([rect2.x2 - rect2.x1, rect2.y2 - rect2.y1])
    cos_theta = math.fabs(np.sum(np.dot(vector_two,vector_one)/(np.absolute(vector_one)*np.absolute(vector_two))))
    if cos_theta > 0.9:
        return True
    return False



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