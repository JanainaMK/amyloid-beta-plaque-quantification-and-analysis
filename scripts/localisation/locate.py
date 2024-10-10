import argparse
import os
import sys
import time

import bioformats
import h5py
import numpy as np
import torch
from skimage.transform import resize

import src.localisation.plaque as p
import src.localisation.plaque_find as pf
import src.localisation.whole_slide_analysis as wsa
import src.util.cli as cli
import src.util.jvm as jvm
from src.data_access import VsiReader
from src.segmentation.evaluation import make_mask_patch_based
from src.segmentation.model import unet


def main():
    start = time.time()
    print("localisation started")

    args = parse_arguments()
    print(args)

    start = args.start
    stop = args.stop
    vsi_folder = args.source_folder

    file_names = os.listdir(vsi_folder)
    vsi_files = [file for file in file_names if file.endswith(".vsi")]
    vsi_files = vsi_files[start:stop]
    print(len(vsi_files), "files found in total")

    print("processing", stop - start, f"slides ({start} - {stop})")

    # file
    vsi_root = args.source_folder
    target_folder = args.target_folder

    # parameters segmentation
    patch_size_segmentation = args.patch_size_segmentation
    downscale_factor = args.downscale_factor
    model_path = args.model_path
    segmentation_setting = args.segmentation_setting

    # parameters localisation
    patch_size_localisation = args.patch_size_localisation
    threshold = int(args.threshold * 255)
    kernel_size = args.kernel_size
    min_size_micron = args.minimum_size
    minsize = int(min_size_micron / 0.274)

    model = initialize_model(model_path, segmentation_setting)
    os.makedirs(target_folder, exist_ok=True)
    jvm.start()
    try:

        for i, vsi_name in enumerate(vsi_files):
            vsi_name = os.path.splitext(vsi_name)[0]
            print("slide", i, ":", vsi_name)

            with h5py.File(f"{target_folder}/{vsi_name}.hdf5", "a") as result_file:
                if "plaques" in result_file:
                    del result_file["plaques"]
                    print("old plaques deleted")

                # this line sets the index of the full size image in the vsi file (see readme).
                full_index = 13 if vsi_name[:5] == "Image" else 0

                print("starting process", time.time() - start)
                raw_reader = bioformats.ImageReader(f"{vsi_root}/{vsi_name}.vsi")

                # Segmentation

                # setup segmentation reader
                vsi_reader = VsiReader(
                    raw_reader,
                    patch_size_segmentation,
                    patch_size_segmentation,
                    downscale_factor,
                    np.uint8,
                    False,
                    True,
                    full_index,
                )
                print("segmentation reader loaded", time.time() - start)
                print("image shape:", vsi_reader.shape)
                print("patchifier shape:", vsi_reader.patch_it.shape)
                grey_matter = segment_gray_matter(
                    segmentation_setting, model, vsi_reader, result_file, vsi_name
                )

                # Localisation

                # setup localisation reader
                vsi_reader = VsiReader(
                    raw_reader,
                    patch_size_localisation,
                    patch_size_localisation,
                    0,
                    np.uint8,
                    False,
                    False,
                    full_index,
                )
                print("localisation reader loaded")
                print("image shape:", vsi_reader.shape)
                print("patchifier shape:", vsi_reader.patch_it.shape)
                plaque_masks = locate_plaques(
                    vsi_reader,
                    grey_matter,
                    threshold,
                    kernel_size,
                    minsize,
                    start,
                    downscale_factor,
                )

                extract_basic_features(result_file, vsi_reader, plaque_masks, start)
    except FileNotFoundError as e:
        raise FileNotFoundError(e)
    finally:
        jvm.stop()


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="Segmentation and Localisation pipeline",
        description="An algorithm that can crop amyloid beta plaques from a vsi file",
    )
    cli.add_io_settings(parser)
    cli.add_segmentation_settings(parser)
    cli.add_localisation_settings(parser)
    cli.add_series_settings(parser)
    parser.add_argument(
        "-ss",
        "--segmentation_setting",
        choices=["no", "load", "create"],
        default="create",
        help="The way the grey matter segmentation is handled. \n 'no' skips the segmentation \n "
        "'load' loads the segmentation from the result file located in the target_folder \n "
        "'create' creates a new segmentation and stores it in the target_folder.",
    )

    return parser.parse_args()


def initialize_model(model_path, segmentation_setting):
    cuda_available = torch.cuda.is_available()
    if segmentation_setting == "create" and cuda_available:
        print("cuda:", cuda_available)
        return unet(model_path)
    elif segmentation_setting == "create" and not cuda_available:
        print("GPU requested, but CUDA is not available, exiting.")
        sys.exit()
    else:
        return None


def segment_gray_matter(segmentation_setting, model, vsi_reader, result_file, vsi_name):
    if segmentation_setting == "no":
        grey_matter = None
    elif segmentation_setting == "create":
        with torch.no_grad():
            grey_matter = (
                make_mask_patch_based(model, vsi_reader)
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )
            if "grey-matter" in result_file:
                del result_file["grey-matter"]
            result_file.create_dataset("grey-matter", data=grey_matter)
            print("grey matter segmented")
    elif "grey-matter" not in result_file:
        raise FileNotFoundError(f"there is no gray matter found for {vsi_name}")
    else:
        grey_matter = result_file["grey-matter"][()]
        print("grey matter loaded")
    print()
    return grey_matter


def locate_plaques(
    vsi_reader, grey_matter, threshold, kernel_size, minsize, start, downscale_factor
):
    pms = []
    for i in range(len(vsi_reader)):
        patch = vsi_reader[i]
        r, c = vsi_reader.patch_it[i]
        h = min(vsi_reader.shape[0] - r, vsi_reader.patch_size)
        w = min(vsi_reader.shape[1] - c, vsi_reader.patch_size)
        print(
            "analyzing patch",
            f"{i}/{len(vsi_reader)}, loc:({r}, {c}), shape: ({h}, {w})",
        )

        gm = wsa.get_grey_matter(r, c, h, w, grey_matter, downscale_factor)
        if gm.mean() < 0.01:
            print("no grey matter")
            print()
            continue
        gm = resize(gm, (h, w))
        print("grey matter found", time.time() - start)

        binary, t = pf.dab_threshold_otsu(patch, threshold)
        binary = gm * binary
        closed = pf.closing(binary, kernel_size)
        plaques, mask = pf.find_plaques(closed, minsize, True, True)
        print(len(plaques), "plaques found", time.time() - start)
        print("threshold used:", t)
        print()

        for pm in plaques:
            x = c + pm.bb.x
            y = r + pm.bb.y
            final_pm = p.PlaqueMask(pm.mask, p.BoundingBox(x, y, pm.bb.w, pm.bb.h))
            pms.append(final_pm)
    return pms


def extract_basic_features(result_file, vsi_reader, pms, start):
    print("slide analyzed in", time.time() - start, "seconds")
    print()
    print("initial detections:", len(pms))
    print("merging...")
    pairs = wsa.get_bordering_detections_from_plaque_mask(pms)
    print(len(pairs), "bordering detections", time.time() - start)
    ccs = wsa.get_connected_components(pairs, len(pms))
    print("detections after merging:", len(ccs), time.time() - start)
    wsa.merge_plaque_masks_from_list(pms, ccs)
    print("detections merged")

    # result
    print("saving detections...")
    n = len(pms)
    plaque_group = result_file.create_group("plaques")
    plaque_group.attrs["length"] = n
    bbs = np.zeros((n, 4), dtype=int)
    areas = np.zeros(n)
    roundnesses = np.zeros(n)
    for i, pm in enumerate(pms):
        plaque_img = vsi_reader.image_reader.rdr.openBytesXYWH(
            0, pm.bb.x, pm.bb.y, pm.bb.w, pm.bb.h
        )
        plaque_img = plaque_img.reshape((pm.bb.h, pm.bb.w, 3))

        bbs[i] = [pm.bb.x, pm.bb.y, pm.bb.w, pm.bb.h]
        areas[i] = np.sum(pm.mask) * 0.274**2  # micron^2
        roundnesses[i] = p.roundness(pm.mask)

        plaque_entry = plaque_group.create_group(str(i))
        plaque_entry.create_dataset("plaque", data=plaque_img)
        plaque_entry.create_dataset("mask", data=pm.mask)
        plaque_entry.attrs["x"] = pm.bb.x
        plaque_entry.attrs["y"] = pm.bb.y
        plaque_entry.attrs["w"] = pm.bb.w
        plaque_entry.attrs["h"] = pm.bb.h
        plaque_entry.attrs["area"] = areas[i]
        plaque_entry.attrs["roundness"] = roundnesses[i]

    if "bbs" in result_file:
        del result_file["bbs"]
    if "area" in result_file:
        del result_file["area"]
    if "roundness" in result_file:
        del result_file["roundness"]
    result_file.create_dataset("bbs", data=bbs)
    result_file.create_dataset("area", data=areas)
    result_file.create_dataset("roundness", data=roundnesses)

    gm = result_file["grey-matter"][()]
    gm_area = np.sum(gm) * (0.274 * 16) ** 2  # micron^2
    gm_area = gm_area / 1000000  # mm^2

    result_file.attrs["n_plaques"] = n
    result_file.attrs["ppmm"] = n / gm_area

    print("plaques saved")


if __name__ == "__main__":
    main()
