import cv2
import numpy as np

from pprint import pprint
from statistics import median
from sys import exit


def hough_transform(edges, img=None, thresh=110):
    """Get contour lines of the receipt in polar coords(r, t)
    using a thresh for the hough accumulator and calculate cartesian coords(x, y).
    """

    lines = cv2.HoughLines(edges, 1, np.pi/180, thresh)

    for x in range(0, len(lines)):
        for rho, theta in lines[x]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            # Draw the line if the img parameter is given
            if img is not None:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return lines


def get_median_lines(intersections, width, height, img):
    """Calculate the median lines from intersections lines
    sorting the intersections list in 4 groups for x(x1,..,x4)
    and 4 groups for y(y1,..,y4).
    """

    jmp_val = 50 # value used to differentiate groups
    first = second = first_p = second_p = third_p = forth_p = None

    intersections.sort(key = lambda x: x[1])
    pprint(intersections)
    for i, x in enumerate(intersections):
        if i+1<len(intersections):
            if intersections[i][0]>width or intersections[i][1]>height:
                continue

            if (intersections[i][1] + jmp_val) <= intersections[i+1][1]:
                first = intersections[:i+1]
                second = intersections[i+1:]

    if first is None or second is None:
        print('Error: must adjust jmp_val or thresh from hough_transform')
        exit()

    first.sort(key = lambda x: x[0])
    for i, x in enumerate(first):
        if i+1<len(first):
            if (first[i][0] + jmp_val) <= first[i+1][0]:
                first_p = first[:i+1]
                second_p = first[i+1:]

    second.sort(key = lambda x: x[0])
    for i, x in enumerate(second):
        if i+1<len(second):
            if (second[i][0] + jmp_val) <= second[i+1][0]:
                third_p = second[:i+1]
                forth_p = second[i+1:]

    if all([first_p, second_p, third_p, forth_p]) is False:
        print('Error: must adjust jmp_val or thresh from hough_transform')
        exit()

    x1 = int(median([p[0] for p in first_p]))
    x2 = int(median([p[0] for p in second_p]))
    x3 = int(median([p[0] for p in third_p]))
    x4 = int(median([p[0] for p in forth_p]))


    # print(x1, x2, x3, x4)
    first.sort(key = lambda x: x[1])
    for i, x in enumerate(first):
        if i+1<len(first):
            if (first[i][0] + jmp_val) <= first[i+1][0]:
                first_p = first[:i+1]
                second_p = first[i+1:]

    second.sort(key = lambda x: x[1])
    for i, x in enumerate(second):
        if i+1<len(second):
            if (second[i][1] + jmp_val) <= second[i+1][1]:
                third_p = second[:i+1]
                forth_p = second[i+1:]

    y1 = int(median([p[1] for p in first_p]))
    y2 = int(median([p[1] for p in second_p]))
    y3 = int(median([p[1] for p in third_p]))
    y4 = int(median([p[1] for p in forth_p]))


    # print(y1, y2, y3, y4)

    img[int(y1), int(x1)] = [0, 255, 0]
    img[int(y2), int(x2)] = [0, 255, 0]
    img[int(y3), int(x3)] = [0, 255, 0]
    img[int(y4), int(x4)] = [0, 255, 0]

    max_x = max([x1, x2, x3, x4])
    max_y = max([y1, y2, y3, y4])
    points_a = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    points_b = np.float32([[0, 0], [max_x, 0], [0, max_y], [max_x, max_y]])

    M = cv2.getPerspectiveTransform(points_a, points_b)
    warped = cv2.warpPerspective(img, M, (x4, y4))

    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.line(img, (x1, y1), (x3, y3), (255, 0, 0), 2)
    cv2.line(img, (x2, y2), (x4, y4), (255, 0, 0), 2)
    cv2.line(img, (x3, y3), (x4, y4), (255, 0, 0), 2)

    return warped


def bradley_roth(image, s=None, t=None):
    """Credits: https://stackoverflow.com/a/33092928
    Numpy implementation of Bradley-Roth adaptive threshold
    """

    img = image.astype(np.float)

    if s is None:
        s = np.round(img.shape[1]//8)

    if t is None:
        t = 15.0

    integral_img = np.cumsum(np.cumsum(img, axis=1), axis=0)

    (rows, cols) = img.shape[:2]
    (X, Y) = np.meshgrid(np.arange(cols), np.arange(rows))

    X = X.ravel()
    Y = Y.ravel()

    s = s + np.mod(s, 2)

    x1 = X - s//2
    x2 = X + s//2
    y1 = Y - s//2
    y2 = Y + s//2

    x1[x1 < 0] = 0
    x2[x2 >= cols] = cols-1
    y1[y1 < 0] = 0
    y2[y2 >= rows] = rows-1

    count = (x2 - x1) * (y2 - y1)

    f1_x = x2
    f1_y = y2
    f2_x = x2
    f2_y = y1 - 1
    f2_y[f2_y < 0] = 0
    f3_x = x1-1
    f3_x[f3_x < 0] = 0
    f3_y = y2
    f4_x = f3_x
    f4_y = f2_y

    sums = integral_img[f1_y, f1_x] - integral_img[f2_y, f2_x] - integral_img[f3_y, f3_x] + integral_img[f4_y, f3_x]

    out = np.ones(rows*cols, dtype=np.bool)
    out[img.ravel()*count <= sums*(100.0 - t)/100.0] = False

    out = 255*np.reshape(out, (rows, cols)).astype(np.uint8)

    return out

