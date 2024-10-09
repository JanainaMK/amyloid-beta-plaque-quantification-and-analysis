import time

import h5py
import numpy as np
from skimage.transform import resize

import src.localisation.plaque as p
import src.localisation.plaque_find as pf
from src.data_access import VsiReader


def get_grey_matter(
    row_pixel, column_pixel, height, width, grey_matter, downscale_factor
):
    r = int(row_pixel / downscale_factor)
    c = int(column_pixel / downscale_factor)
    h = int(height / downscale_factor)
    w = int(width / downscale_factor)
    shape = grey_matter.shape
    if r >= shape[0]:
        r = shape[0] - 1
    if c >= shape[1]:
        c = shape[1] - 1
    if h == 0:
        h = 1
    if w == 0:
        w = 1
    if grey_matter is None:
        return np.ones((h, w), bool)
    res = grey_matter[r : r + h, c : c + w]
    return res


def find_global_threshold(reader, grey_matter=None, downscale=None):
    thresholds = np.zeros(len(reader))
    start = time.time()
    for i in range(len(reader)):
        if grey_matter is not None:
            r, c = reader.patch_it[i]
            gm = get_grey_matter(r, c, reader.patch_size_localisation, downscale)
            if gm.mean() < 0.01:
                continue
        patch = reader[i]
        _, t = pf.dab_threshold_otsu(patch, 0)
        print(f"patch {i}/{len(reader)}", time.time() - start)
        thresholds[i] = t
    return thresholds.max()


def get_detections(
    vsi_reader: VsiReader,
    grey_matter: np.ndarray,
    threshold: int,
    use_otsu: bool,
    kernel_size: int,
    minimum_size: int,
    downscale_factor: int,
):
    start = time.time()
    bbs = []
    for i in range(len(vsi_reader)):
        patch = vsi_reader[i]
        r, c = vsi_reader.patch_it[i]
        h = min(vsi_reader.shape[0] - r, vsi_reader.patch_size)
        w = min(vsi_reader.shape[1] - c, vsi_reader.patch_size)
        print(
            "analysing patch",
            f"{i}/{len(vsi_reader)}, loc:({r}, {c}), shape: ({h}, {w}), alt shape:",
            patch.shape,
        )

        gm = get_grey_matter(r, c, h, w, grey_matter, downscale_factor)
        if gm.mean() < 0.01:
            print("no grey matter", time.time() - start)
            print()
            continue
        gm = resize(gm, (h, w))
        print("grey matter found", time.time() - start)

        binary, t = (
            pf.dab_threshold_otsu(patch, threshold)
            if use_otsu
            else pf.dab_threshold(patch, threshold)
        )
        closed = gm * pf.closing(binary, kernel_size)
        plaques, mask = pf.find_plaques(closed, minimum_size, True, True)
        print(len(plaques), "plaques found", time.time() - start)
        print("threshold used:", t)
        print()

        for plaque in plaques:
            x = c + plaque.bb.x
            y = r + plaque.bb.y
            bbs.append(p.BoundingBox(x, y, plaque.bb.w, plaque.bb.h))
    print("slide analysed in", time.time() - start, "seconds")
    return bbs


def get_bordering_detections(bbs: list, iou_threshold=0):
    n = len(bbs)
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            x_overlap = min(bbs[i].x + bbs[i].w, bbs[j].x + bbs[j].w) - max(
                bbs[i].x, bbs[j].x
            )
            y_overlap = min(bbs[i].y + bbs[i].h, bbs[j].y + bbs[j].h) - max(
                bbs[i].y, bbs[j].y
            )
            if x_overlap > 0 and y_overlap == 0:
                x_range = max(bbs[i].x + bbs[i].w, bbs[j].x + bbs[j].w) - min(
                    bbs[i].x, bbs[j].x
                )
                iou = x_overlap / x_range
            elif y_overlap > 0 and x_overlap == 0:
                y_range = max(bbs[i].y + bbs[i].w, bbs[j].y + bbs[j].w) - min(
                    bbs[i].y, bbs[j].y
                )
                iou = y_overlap / y_range
            else:
                iou = 0
            if iou > iou_threshold:
                edge_list.append((i, j))
    return edge_list


def get_bordering_detections_from_plaque_mask(pms: list, iou_threshold=0):
    n = len(pms)
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            x_overlap = min(pms[i].bb.x + pms[i].bb.w, pms[j].bb.x + pms[j].bb.w) - max(
                pms[i].bb.x, pms[j].bb.x
            )
            y_overlap = min(pms[i].bb.y + pms[i].bb.h, pms[j].bb.y + pms[j].bb.h) - max(
                pms[i].bb.y, pms[j].bb.y
            )
            if x_overlap > 0 and y_overlap == 0:
                x_range = max(
                    pms[i].bb.x + pms[i].bb.w, pms[j].bb.x + pms[j].bb.w
                ) - min(pms[i].bb.x, pms[j].bb.x)
                iou = x_overlap / x_range
            elif y_overlap > 0 and x_overlap == 0:
                y_range = max(
                    pms[i].bb.y + pms[i].bb.w, pms[j].bb.y + pms[j].bb.w
                ) - min(pms[i].bb.y, pms[j].bb.y)
                iou = y_overlap / y_range
            else:
                iou = 0
            if iou > iou_threshold:
                edge_list.append((i, j))
    return edge_list


def get_connected_components(edge_list, n):
    visited = np.zeros(n, bool)
    groups = []
    for v in range(n):
        if visited[v]:
            continue
        cc = visit(v, [], edge_list, visited)
        groups.append(cc)
    return groups


def visit(v, cc, edge_list, visited):
    cc.append(v)
    visited[v] = True
    for i, j in edge_list:
        if v == i and not visited[j]:
            cc = visit(j, cc, edge_list, visited)
        elif v == j and not visited[i]:
            cc = visit(i, cc, edge_list, visited)
    return cc


def merge_bounding_boxes(bbs, ccs):
    n = len(bbs)
    to_delete = np.zeros(n, bool)
    for cc in ccs:
        if len(cc) == 1:
            continue
        min_x, min_y, max_x, max_y = 10**6, 10**6, 0, 0
        for i in cc:
            to_delete[i] = True
            min_x = min(bbs[i].x, min_x)
            min_y = min(bbs[i].y, min_y)
            max_x = max(bbs[i].x + bbs[i].w, max_x)
            max_y = max(bbs[i].y + bbs[i].h, max_y)
        bbs.append(p.BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y))

    for i in range(n - 1, -1, -1):
        if to_delete[i]:
            del bbs[i]
    return bbs


def merge_plaque_masks_from_list(pms, ccs):
    n = len(pms)
    to_delete = np.zeros(n, bool)
    for cc in ccs:
        if len(cc) == 1:
            continue

        min_x, min_y, max_x, max_y = 10**6, 10**6, 0, 0
        for i in cc:
            to_delete[i] = True
            min_x = min(pms[i].bb.x, min_x)
            min_y = min(pms[i].bb.y, min_y)
            max_x = max(pms[i].bb.x + pms[i].bb.w, max_x)
            max_y = max(pms[i].bb.y + pms[i].bb.h, max_y)

        merged_bb = p.BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)
        merged_mask = np.zeros((max_y - min_y, max_x - min_x), bool)
        for i in cc:
            x = pms[i].bb.x - min_x
            y = pms[i].bb.y - min_y
            merged_mask[y : y + pms[i].bb.h, x : x + pms[i].bb.w] = pms[i].mask
        pms.append(p.PlaqueMask(merged_mask, merged_bb))

    for i in range(n - 1, -1, -1):
        if to_delete[i]:
            del pms[i]
    return pms


def merge_plaque_masks(plaques: h5py.Group, ccs: list):
    indices = [eval(i) for i in plaques.keys()]
    current_i = max(indices) + 1
    for cc in ccs:
        if len(cc) == 1:
            continue
        min_x, min_y, max_x, max_y = 10**6, 10**6, 0, 0
        pms = p.read_plaque_masks(plaques, cc)
        for pm in pms:
            min_x = min(pm.bb.x, min_x)
            min_y = min(pm.bb.y, min_y)
            max_x = max(pm.bb.x + pm.bb.w, max_x)
            max_y = max(pm.bb.y + pm.bb.h, max_y)
        merged_bb = p.BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)
        merged_mask = p.PlaqueMask(
            np.zeros((max_y - min_y, max_x - min_x), bool), merged_bb
        )
        for pm in pms:
            x = pm.bb.x - min_x
            y = pm.bb.y - min_y
            merged_mask.mask[y : y + pm.bb.h, x : x + pm.bb.w] = pm.mask
        p.save_plaque_mask(plaques, merged_mask, current_i)
        current_i += 1
        for i in cc:
            del plaques[str(i)]
