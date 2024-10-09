import cv2
import numpy as np
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity

from src.localisation import BoundingBox, PlaqueMask


def dab_threshold(image: np.ndarray, threshold: int, rescale: bool = False):
    temp = (rgb2hed(image)[:, :, 2] * 255).astype(np.uint8)
    if rescale:
        temp = rescale_intensity(temp, (temp.min(), temp.max()), (0, 255)).astype(
            np.uint8
        )
    retval, temp = cv2.threshold(temp, threshold, 255, cv2.THRESH_BINARY)
    return temp, retval


def dab_threshold_otsu(image: np.ndarray, min_threshold: int, rescale: bool = False):
    dab = (rgb2hed(image)[:, :, 2] * 255).astype(np.uint8)
    if rescale:
        dab = rescale_intensity(dab, (dab.min(), dab.max()), (0, 255)).astype(np.uint8)
    retval, thresh = cv2.threshold(dab, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if retval < min_threshold:
        retval, thresh = cv2.threshold(dab, min_threshold, 255, cv2.THRESH_BINARY)
    return thresh, retval


def closing(binary: np.ndarray, kernel_size: int):
    kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if kernel_size > 0
        else None
    )
    temp = (
        cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        if kernel is not None
        else binary
    )
    return temp


def find_plaques(binary, minsize, return_masks, use_area=False):
    min_area = (minsize / 2) ** 2 * np.pi
    num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, 4
    )
    result = []
    full_mask = binary.copy().astype(bool)
    for i in range(1, num_components):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        if (use_area and a < min_area) or (not use_area and max(h, w) < minsize):
            full_mask[labels == i] = 0
            continue
        bb = BoundingBox(x, y, w, h)
        if return_masks:
            # mask = labels[y:y + h, x:x + w].astype(bool)
            mask = labels[y : y + h, x : x + w] == i
            result.append(PlaqueMask(mask, bb))
        else:
            result.append(bb)
    return result, full_mask
