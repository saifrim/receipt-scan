import cv2
from numpy import pi, cos, sin
from pprint import pprint
from statistics import median
from sys import exit


def hough_transform(edges, img=None, thresh=110):
    """Get contour lines of the receipt in polar coords(r, t)
    using a thresh for the hough accumulator and calculate cartesian coords(x, y).
    """

    lines = cv2.HoughLines(edges, 1, pi/180, thresh)

    for x in range(0, len(lines)):
        for rho, theta in lines[x]:
            a = cos(theta)
            b = sin(theta)
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

    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.line(img, (x1, y1), (x3, y3), (255, 0, 0), 2)
    cv2.line(img, (x2, y2), (x4, y4), (255, 0, 0), 2)
    cv2.line(img, (x3, y3), (x4, y4), (255, 0, 0), 2)

