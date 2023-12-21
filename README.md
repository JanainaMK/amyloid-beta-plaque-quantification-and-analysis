# Amyloid-beta plaque quantification and analysis
The project consists of a machine learning pipeline designed to automatically locate and categorize Amyloid-Beta (AB) plaques into different groups, based on their formation types (compact, diffuse, etc.). This is a thesis project taken on by Master student Janaína Moreira-Kanaley, which expands on the [code base](https://github.com/Gjel/amyloid-beta-plaque-quantification-and-analysis/tree/master) and [paper](https://repository.tudelft.nl/islandora/object/uuid:7b67aaa9-41a5-4bc3-b987-68f34ed145d9?collection=education) by Chiel de Vries. The repository contains the code necessary to reproduce the results of the paper, but not the dataset. The dataset contains personally identifiable data, therefore it cannot be shared publicly. 

## Installation
To be able to run the code, several packages should be installed after cloning or forking the project. This can be done in two ways: Anaconda and pip.
> :warning: Both methods result in slightly different environments. both methods exist because the anaconda version was necessary for the HPC, but it didn't work on my local machine. The pip version does, but is slightly different as a result. The code was tested on both versions and should run fine.

### Anaconda 
To run the code on the TU Delft HPC, it is best to use anaconda for installation. The `Abeta.yml` file contains all the necessary packages so running `conda env create -f Abeta.yml` should install the environment correctly. From here the code can be run.

### Pip
If you prefer pip, there is also a `requirements.txt` file that can be used to install the packages. After creating your virtual environment, run `pip install -r requirements.txt` to install all packages. 

## Pipeline
The pipeline can be divided into four different stages

1. **Data Preparation**
2. **Segmentation**
3. **Localisation**
4. **Categorisation**


To start the pipeline run the `pipeline.sh` script.
```
sh pipeline.sh
```

Optionally, follow the step-by-step instructions below. For the steps described, use `--help` command for more detailed instructions on the available parameters for each python script.
  
### 1. Data Preparation
The Whole Slide Images (WSIs) of the Amyloid-Beta plaques are stored as `.vsi` files. Expert defined labels for the grey matter are saved as `.hdf5` files. The data preparation stage of the pipeline downsamples the images and labels, and saves them in hdf5 format. It additionally creates an overarching hdf5 file that links the hdf5 labels to their correspoding images.

To run this stage of the pipeline:
```
sh scripts/segmentation/prepare_data
```

### 2. Segmentation
During this phase of the pipeline, a model is trained to produce pixel-level predictions for Whole Slide Images (WSIs) that indicate whether a pixel corresponds to grey matter or not. The training process utilizes the Centurian dataset annotated by experts. The trained model can be applied to segment the grey matter from the background within the WSIs.

To start the training:
```
python -u scripts/segmentation/train.py
```

To generate predictions for grey matter segmentation:
```
python -u scripts/segmentation/segment.py
```

### 3. Localisation
This phase locates the AB plaques in the WSIs. During this process, it generates masks and bounding boxes for each located plaque. The segmented grey matter is used to effectively prune plaque search locations in the images. Basic features of the plaques are also computed here: `area`, `roundness`.

To run localisation:
```
python -u scripts/localisation/locate.py -sf <SOURCE_FOLDER> -tf <TARGET_FOLDER>
```


### 4. Categorisation
In this phase of the pipeline, a K-Means model is trained on the extracted features, which is used to categorize the plaques into different groups. These features include `area`, `roundness`, and 1000 features resulting from applying `AlexNet` to the bounding box image of each plaque. There is the option to specify the combination of features, `alex_features` or `simple_features` (`area` and `roundness`).

To compute AlexNet features:
```
python -u scripts/categorisation/features.py
```

To train cluster model and/or generate cluster predictions:
```
python -u scripts/categorisation/cluster.py --alex_features --simple_features --fit --predict -noc <number_of_clusters>
```

## Tips: Grey matter segmentation and plaque localisation pipeline
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
