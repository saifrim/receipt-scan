import cv2

from binarization import get_roi

MAGNITUDE = 1.5


if __name__ == '__main__':
    img = cv2.imread('img/receipt-best.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    width, height = int(w*MAGNITUDE), int(h*MAGNITUDE)

    roi = get_roi(img_gray)

    cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Input', width, height)

    cv2.imshow('Input', img)
    cv2.imshow('Output', roi)

    cv2.waitKey(0)

