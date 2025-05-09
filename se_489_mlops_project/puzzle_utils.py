import numpy as np
import cv2
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from numpy.typing import NDArray

def order_points(pts: NDArray[np.float32]) -> NDArray[np.float32]:
    rect: NDArray[np.float32] = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: NDArray[np.uint8], pts: NDArray[np.float32]) -> NDArray[np.uint8]:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped: NDArray[np.uint8] = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def find_puzzle(image: NDArray[np.uint8], debug: bool = False) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    if debug:
        plt.figure(figsize=(18, 18))
        plt.subplot(221)
        plt.axis('off')
        plt.title("Thresholded image")
        plt.imshow(thresh, cmap='gray')

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    puzzleCnt: Optional[NDArray[np.int32]] = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            puzzleCnt = approx
            break

    if puzzleCnt is None:
        raise Exception("Could not find the puzzle outline.")

    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        plt.subplot(222)
        plt.axis('off')
        plt.title("Puzzle Outline")
        plt.imshow(output, cmap='gray')

    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2).astype(np.float32))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2).astype(np.float32))

    if debug:
        plt.subplot(223)
        plt.axis('off')
        plt.title("Original warped")
        plt.imshow(puzzle, cmap='gray')

        plt.subplot(224)
        plt.axis('off')
        plt.title("Gray warped")
        plt.imshow(warped, cmap='gray')

    return puzzle, warped

def extract_digit(cell: NDArray[np.uint8], debug: bool = False) -> Optional[NDArray[np.uint8]]:
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    if debug:
        plt.figure(figsize=(12, 12))
        plt.subplot(121)
        plt.title("Thresholded Image")
        plt.axis('off')
        plt.imshow(thresh, cmap='gray')

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None

    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype='uint8')
    cv2.drawContours(mask, [c], -1, 255, -1)

    (h, w) = thresh.shape
    percent_filled = cv2.countNonZero(mask) / float(w * h)

    if percent_filled < 0.03:
        return None

    digit: NDArray[np.uint8] = cv2.bitwise_and(thresh, thresh, mask=mask)


    if debug:
        plt.subplot(122)
        plt.title("Digit")
        plt.axis('off')
        plt.imshow(digit, cmap='gray')

    return digit
