import cv2
import numpy as np

from sys import argv, exit

from kmeans import segment_by_angle_kmeans, segmented_intersections
from preprocess import hough_transform, get_median_lines


if __name__ == '__main__':
    if len(argv) == 1:
        print('Image argument not provided!')
        exit()

    img = cv2.imread('img/{}'.format(argv[1]))
    if img is None:
        print('Wrong image format!')
        exit()

    height, width, _ =  img.shape
    if height > 640 or width > 480:
        img = cv2.resize(img, (0, 0), fx=640/height, fy=480/width)

    print(img.shape)

    # Random forest based model pretrained available in opencv_contrib lib
    edge_detector = cv2.ximgproc.createStructuredEdgeDetection('model.yml')

    rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edges = edge_detector.detectEdges(np.float32(rgb_im) / 255.0)

    print(edges.shape)

    orimap = edge_detector.computeOrientation(edges)
    edges = edge_detector.edgesNms(edges, orimap)

    edges = (edges * 255).round().astype(np.uint8)

    lines = hough_transform(edges)

    segmented = segment_by_angle_kmeans(lines)
    intersections = segmented_intersections(segmented)
    get_median_lines(intersections, width, height, img)

    cv2.imshow('Edged image', edges)
    cv2.imshow('Output', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

