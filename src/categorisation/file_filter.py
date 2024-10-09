import os
import re

keywords_frontal = ["F2", "F2A", "F2B", "frontalA"]  # frontal
keywords_parietal = ["LPI", "lobparinf", "lobparinf1", "lobparinfA"]  # parietal
keywords_temporal = ["Tpole", "temppole", "temppoleA", "temppool"]  # temporal
keywords_occipital = ["Occipital", "occipital", "occipitalA"]  # occipital
exclude_file = "Image_2021-065_frontalA_AB4"


def get_brain_name(file_name):
    file_name = file_name.split(".")[0]
    pattern = r"\d+_(.*)"  # regex pattern should match on <number>_x
    match = re.search(pattern, file_name)

    if match:
        return match.group(1).split("_")[0]
    else:
        return None


def get_dataset_name(file_name, cent_path, ad_path):
    cent_file_names = [
        os.path.splitext(file)[0]
        for file in os.listdir(cent_path)
        if file.endswith(".vsi")
    ]
    ad_file_names = [
        os.path.splitext(file)[0]
        for file in os.listdir(ad_path)
        if file.endswith(".vsi")
    ]
    flag = 0
    if file_name in cent_file_names:
        flag += 1
        res = "100+"
    elif file_name in ad_file_names:
        flag += 1
        res = "AD"
    elif flag != 1:
        raise Exception(f"{file_name} does not belong to any existing dataset.")
    return res


def get_region_name(file_name):
    if exclude_file in file_name:
        raise Exception(f"{file_name} was set to be excluded.")
    name = get_brain_name(file_name)
    flag = 0
    if name in keywords_frontal:
        flag += 1
        res = "Frontal"
    elif name in keywords_parietal:
        flag += 1
        res = "Parietal"
    elif name in keywords_temporal:
        flag += 1
        res = "Temporal"
    elif name in keywords_occipital:
        flag += 1
        res = "Occipital"
    if flag == 0:
        raise Exception(f"{name} does not belong to any cerebral region.")
    if flag > 1:
        raise Exception(f"{name} belongs to multiple cerebral regions.")
    return res


def valid_region(file_name):
    if exclude_file in file_name:
        return False
    name = get_brain_name(file_name)
    flag = 0
    if name in keywords_frontal:
        flag += 1
    elif name in keywords_parietal:
        flag += 1
    elif name in keywords_temporal:
        flag += 1
    elif name in keywords_occipital:
        flag += 1
    if flag > 1:
        raise Exception(f"{name} belongs to multiple cerebral regions.")
    return flag == 1


def get_participant_number(file_name):
    pattern = r"(\d+-\d+)"  # regex pattern should match on <number>-<number>
    match = re.search(pattern, file_name)
    if match:
        return match.group(1)
    else:
        return None
