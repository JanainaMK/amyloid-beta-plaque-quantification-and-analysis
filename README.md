# Amyloid-beta plaque quantification and analysis
This repository houses the code base that accompanies the paper [Amyloid-beta plaque quantification and analysis](https://repository.tudelft.nl/islandora/object/uuid:7b67aaa9-41a5-4bc3-b987-68f34ed145d9?collection=education) by Chiel de Vries. This work was done as the graduation project of the author, who graduated successfully from The TU Delft. The repository contains the code necessary to reproduce the results of the paper, but not the dataset. The dataset contains personally identifiable data, therefore it cannot be shared publicly. 

## Installation
To be able to run the code, several packages should be installed after cloning or forking the project. This can be done in two ways: Anaconda and pip.
> :warning: Both methods result in slightly different environments. both methods exist because the anaconda version was necessary for the HPC, but it didn't work on my local machine. The pip version does, but is slightly different as a result. The code was tested on both versions and should run fine.

### Anaconda 
To run the code on the TU Delft HPC, it is best to use anaconda for installation. The `Abeta.yml` file contains all the necessary packages so running `conda env create -f Abeta.yml` should install the environment correctly. From here the code can be run.

### Pip
If you prefer pip, there is also a `requirements.txt` file that can be used to install the packages. After creating your virtual environment, run `pip install -r requirements.txt` to install all packages. 

## Grey matter segmentation and plaque localisation pipeline
The most important file in this codebase is `script/localisation/loc.py`. This script runs the grey matter segmentation and plaque localisation steps for a single `vsi` file. It can be run by calling `python script/localisation/loc.py <VSI_FILE_NAME> --source_folder <PATH_TO_DATA_FOLDER> --target_folder <PATH_TO_RESULT_FOLDER>`, which will run it with the default settings (the final settings used in the paper). All other settings can be changed using extra arguments, run `python script/localisation/loc.py --help` for more information. 

> :warning: Due to quirks in the `vsi` file format, the script might not select the correct series in the file. The file stores the image in a pyramidal format where each level of the pyramid is a 2x downsampled version of the previous. The grey matter segmentation uses the 4th level (16x downsampled) and the plaque localisation uses the 0th level (full resolution). However, some `vsi` files have extra data attached to them, meaning that the index of the full resolution image is not located at the first series. To adapt this you can edit the `full_index` variable in the `loc.py` file and set it t the index used by your vsi file.  

This script creates a result file in the target folder. This file is stored in HDF5 format and has the following structure:
- __'root'__: the root of the file. __Attributes__: _n_plaques_: The number of plaques found in the image. _ppmm_: The average number of plaques per square millimeter of grey matter in the image.  
  - __area__: An array with the area of all plaques found in this image in square micrometers.
  - __bbs__: An array with the bounding boxes of all plaques found in the image. It is of size nx4 where the second dimension represents the x and y pixel coordinates of the top left corner of the bounding box and the width and height of the bounding box in pixels respectively.
  - __grey-matter__: An array that contains the grey matter segmentation. This array is downsampled 16 times compared to the original image
  - __roundness__: An array with the roundness of all plaques found in this image.
  - __plaques__: A group containing all the plaques found in the image. __Attributes__: _length_: the number of plaques in the image.
    - __0..n__: A group for each plaque in the image. Each plaque is labeled with an index. __Attributes__: _area_: The area of the plaque in square micrometers. _roundness_: The roundness of the plaque. _x_: The x coordinate of the top left of the bounding box around the plaque in pixels. _y_: The y coordinate of the top left of the bounding box around the plaque in pixels. _w_: The width of the bounding box around the plaque in pixels. _h_: The height of the bounding box around the plaque in pixels.
      - __mask__: An array containing the mask that covers the plaque.
      - __plaque__: An array that contains an image of the plaque in RGB. It is cropped from the original image using the bounding box. 

> :warning: For all the results that are given in micrometers or millimeters the code assumes that each pixel in the original image is 0.274 by 0.274 micrometers.

## Usable scripts
This is a list of scripts that are updated and cleaned up. They should be runnable without much issue and have some documentation in the form of a `--help` command. 
- localisation
  - loc.py: runs the segmentation and localisation pipeline for a single image file.
  - loc_series.py: runs the segmentation and localisation pipeline for a series of image files in the same folder.
- segmentation:
  - segment.py: runs the segmentation pipeline for a single image file.
  - loc_series.py: runs the segmentation pipeline for a series of image files in the same folder.

This list was last updated on November 21st 2023


