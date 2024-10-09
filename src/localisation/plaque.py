from collections import namedtuple

import cv2
import h5py
import numpy as np

BoundingBox = namedtuple("BoundingBox", ["x", "y", "w", "h"])
PlaqueMask = namedtuple("PlaqueMask", ["mask", "bb"])


def bounding_boxes_to_numpy(bbs):
    res = np.zeros((len(bbs), 4), int)
    for i, bb in enumerate(bbs):
        res[i] = (bb.x, bb.y, bb.w, bb.h)
    return res


def numpy_to_bounding_boxes(arr):
    res = []
    for bb in arr:
        res.append(BoundingBox(bb[0], bb[1], bb[2], bb[3]))
    return res


def save_plaque_mask(plaques: h5py.Group, mask: PlaqueMask, name):
    if not isinstance(name, str):
        name = str(name)
    if name in plaques:
        del plaques[name]
    plaque = plaques.create_group(name)
    plaque.create_dataset("x", data=mask.bb.x)
    plaque.create_dataset("y", data=mask.bb.y)
    plaque.create_dataset("w", data=mask.bb.w)
    plaque.create_dataset("h", data=mask.bb.h)
    plaque.create_dataset("mask", data=mask.mask)


def read_bounding_boxes(plaques: h5py.Group):
    bbs = []
    for plaque in plaques:
        x = plaques[f"{plaque}/x"][()]
        y = plaques[f"{plaque}/y"][()]
        w = plaques[f"{plaque}/h"][()]
        h = plaques[f"{plaque}/w"][()]
        bbs.append(BoundingBox(x, y, w, h))
    return bbs


def read_plaque_masks(plaques: h5py.Group, indices: list = None):
    pms = []
    for i in indices:
        x = plaques[f"{i}/x"][()]
        y = plaques[f"{i}/y"][()]
        w = plaques[f"{i}/w"][()]
        h = plaques[f"{i}/h"][()]
        mask = plaques[f"{i}/mask"][()]
        pms.append(PlaqueMask(mask, BoundingBox(x, y, w, h)))
    return pms


def maximum_inscribed_circle(mask: np.ndarray):
    """Gets the maximum inscribed circle"""
    padded = cv2.copyMakeBorder(
        mask.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0
    )
    dist_map = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius, _, center = cv2.minMaxLoc(dist_map)
    diameter = 2 * radius
    return diameter, center


def minimum_circumscribed_circle(mask: np.ndarray):
    """returns the diameter of the minimum circumscribed circle"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )[-2:]
    center, radius = cv2.minEnclosingCircle(np.concatenate(contours, 0))
    diameter = 2 * radius
    return diameter, center


def roundness(mask: np.ndarray):
    d_in, _ = maximum_inscribed_circle(mask)
    d_cir, _ = minimum_circumscribed_circle(mask)
    return d_in / d_cir
