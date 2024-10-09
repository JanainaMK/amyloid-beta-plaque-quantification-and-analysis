import os

FRONTAL = "Middle frontal gyrus"
TEMP_MID = "Middle temporal gyrus"
TEMP_POLE = "Temporal pole"
PARIETAL = "Parietal lobe"
LOB_PAR_INF = "Inferior parietal lobe"
OCC = "Occipital pole"

REGION_DICT = {
    "F2": FRONTAL,
    "F2A": FRONTAL,
    "F2B": FRONTAL,
    "frontalA": FRONTAL,
    "temp2": TEMP_MID,
    "temp12": TEMP_MID,
    "tempmid": TEMP_MID,
    "tempmidA": TEMP_MID,
    "temppole": TEMP_POLE,
    "temppoleA": TEMP_POLE,
    "temppool": TEMP_POLE,
    "temppoleB": TEMP_POLE,
    "Tpole": TEMP_POLE,
    "Pgrijs": PARIETAL,
    "parietal": PARIETAL,
    "lobparinf": LOB_PAR_INF,
    "lobparinf1": LOB_PAR_INF,
    "lobparinfB": LOB_PAR_INF,
    "lobparinfA": LOB_PAR_INF,
    "LPI": LOB_PAR_INF,
    "occipital": OCC,
    "occipitalA": OCC,
    "occipitalB": OCC,
    "occipitalpoolA": OCC,
    "Occipital": OCC,
    "occlesie": "WRONG",
}


def from_vsi_path(name: str):
    no_path = os.path.basename(name)
    return from_vsi(no_path)


def from_vsi(name: str):
    return Name(name[:-4])


def is_vsi(name: str):
    return name[-4:] == ".vsi"


def from_qupath(name: str):
    no_mag = name.split(" - ")[0]
    return from_vsi(no_mag)


def is_qupath(name: str):
    return name.count("- 20x_BF_01") != 0


def from_base(name: str):
    return Name(name)


def is_base(name: str):
    return not is_vsi(name) and not is_qupath(name)


def from_hdf5(name: str):
    return Name(name[:-5])


class Name:
    def __init__(self, base_name: str):
        self.base = base_name

    def to_vsi(self):
        return f"{self.base}.vsi"

    def to_qupath(self):
        return f"{self.to_vsi()} - 20x_BF_01"

    def to_hdf5(self):
        return f"{self.base}.hdf5"

    def to_base(self):
        return self.base

    def to_individual(self):
        parts = self.base.split("_")
        offset = 1 if parts[0] == "Image" else 0
        return parts[0 + offset]

    def to_region(self):
        parts = self.base.split("_")
        offset = 1 if parts[0] == "Image" else 0
        return REGION_DICT[parts[1 + offset]]
