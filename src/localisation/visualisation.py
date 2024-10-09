from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2hed
from skimage.io import imsave

from src.localisation import PlaqueMask


def draw_bounding_boxes(target, bounding_boxes, color):
    res = target.copy()
    for bb in bounding_boxes:
        if isinstance(bb, PlaqueMask):
            cv2.rectangle(
                res,
                (bb.bb.x, bb.bb.y),
                (bb.bb.x + bb.bb.w, bb.bb.y + bb.bb.h),
                color,
                2,
            )
        else:
            cv2.rectangle(res, (bb.x, bb.y), (bb.x + bb.w, bb.y + bb.h), color, 2)
    return res


def crop_bounding_boxes(source, bounding_boxes, target_path, target_name):
    for entry in bounding_boxes:
        bb = entry.bb if isinstance(entry, PlaqueMask) else entry
        plaque = source[bb.y : bb.y + bb.h, bb.x : bb.x + bb.w, :]
        file_name = f"{target_name}-x{bb.x}-y{bb.y}.png"
        cv2.imwrite(join(target_path, file_name), plaque)


def draw_masks(target, masks, color, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    res = target.copy()
    contours = np.zeros(res.shape[:2])
    for mask in masks:
        temp = np.zeros(contours.shape)
        temp[mask.bb.y : mask.bb.y + mask.bb.h, mask.bb.x : mask.bb.x + mask.bb.w] = (
            mask.mask.astype(np.uint8)
        )
        temp -= cv2.morphologyEx(temp, cv2.MORPH_ERODE, kernel)
        contours[temp > 0] = 1
    res[contours > 0] = color
    return res


def draw_full_mask(target, full_mask, color, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = full_mask.astype(np.uint8)
    contours = mask - cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
    res = target.copy()
    res[contours > 0] = color
    return res


def make_dab_histogram(image: np.ndarray, name: str = "noname"):
    dab = (rgb2hed(image)[:, :, 2] * 255).astype(np.uint8)
    hist, bins = np.histogram(dab.ravel(), 256, [0, 256])
    i = -1
    while hist[i] < 20:
        i -= 1
    print(i)

    plt.figure()  # figsize=(12.8, 9.6))
    plt.title(name)
    plt.bar(range(len(hist[:i])), hist[:i])
    plt.savefig(f"visuals/hist/{name}.png")
    plt.close()


def make_dab_image(image: np.ndarray, name: str = "noname"):
    dab = (rgb2hed(image)[:, :, 2] * 255).astype(np.uint8)
    imsave(f"visuals/dab/{name}.png", dab)
