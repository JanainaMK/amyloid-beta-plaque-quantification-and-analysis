# Amyloid-beta plaque quantification and analysis
The project consists of a pipeline designed to automatically locate and categorise Amyloid-Beta (AB) plaques into different groups based on their formations (compact, diffuse, etc.). This is part of a Master's thesis project completed by Janaína Moreira-Kanaley ([paper](https://resolver.tudelft.nl/uuid:c1f7047c-43aa-4c9a-955a-396ea1f317ad)), which expands on the work of Chiel de Vries ([code base](https://github.com/Gjel/amyloid-beta-plaque-quantification-and-analysis/tree/master), [paper](https://repository.tudelft.nl/islandora/object/uuid:7b67aaa9-41a5-4bc3-b987-68f34ed145d9?collection=education)). The main contribution to the previous work is the categorisation stage of the pipeline. The dataset is not included since it contains sensitive information.

## Installation
To be able to run the code, several packages should be installed after cloning or forking the project. This can be done in two ways: Anaconda and pip. Python version 3.9 was used for this project.
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


To start the pipeline run/modify the `pipeline.sh` script.

> :warning: 
>  If this script is executed, the pipeline will start at the first stage which includes recreating the downsampled VSI files and completing segmentation and localisation again. Run the entire process only if you want to rebuild the entire pipeline from start to finish.


```
sh pipeline.sh
```

Optionally, follow the step-by-step instructions below. This is recommended in cases where only individuals steps of the pipeline are desired. For any steps described, use `--help` command for more detailed instructions on the available parameters for each python script.
  
## 1. Data Preparation
The Whole Slide Images (WSIs) of the AB plaques are stored as `.vsi` files. Labels for the grey matter are saved as `.hdf5` files. The data preparation stage of the pipeline downsamples the images and labels, and saves them in HDF5 format. It additionally creates an overarching HDF5 file that links the HDF5 labels to their corresponding images.

To start the data preparation stage run/configure `scripts/data_preparation/prepare_data`:
```
sh scripts/data_preparation/prepare_data.sh
```

### OR

There is the option to run the Python files individually.
For downsampling the images and saving them in HDF5 format run:
```
python -u scripts/data_preparation/downsample_images.py --save_dir <SAVE_DIR> --vsi_dir <VSI_DIR> -dl <DOWNSAMPLE_LEVEL>
```

For downsampling the grey matter labels and saving them in HDF5 format run:
```
python -u scripts/data_preparation/downsample_labels.py --save_dir <SAVE_DIR> -ld <ORIG_LABELS_DIR> -dl <DOWNSAMPLE_LEVEL>
```

To link downsampled labels and images, and assign train validation test sets for segmentation training, use:
```
python -u scripts/data_preparation/link_images_labels.py --split_data --save_dir <SAVE_DIR>
```

## 2. Grey Matter Segmentation
During this phase of the pipeline, a model is trained to produce pixel-level predictions for the WSIs that determine the presence of grey matter. The trained model can be applied to prune areas of the WSIs for a faster plaque localisation process.

To train the U-Net model for segmentation:
```
python -u scripts/segmentation/train.py
```

The model is saved under the `models/segmentation/unet` directory.

## 3. Plaque Localisation
This phase locates the AB plaques in the grey matter. During this process, it generates masks and bounding box images for each located plaque. The segmented grey matter is used to effectively prune plaque search locations in the images. Some basic features of the plaques are also computed here: `area`, `roundness`.

To run localisation:
```
python -u scripts/localisation/locate.py -sf <SOURCE_VSI_FOLDER> -tf <TARGET_SAVE_HDF5_FOLDER> -mp <MODEL_PATH>
```

The most important file in this stage is `scripts/localisation/locate.py`. This script creates a result file in the target folder. This file is stored in HDF5 format and has the following structure:
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

> :warning: Due to quirks in the `vsi` file format, the script might not select the correct series in the file. The file stores the image in a pyramidal format where each level of the pyramid is a 2x downsampled version of the previous. The grey matter segmentation uses the 4th level (16x downsampled) and the plaque localisation uses the 0th level (full resolution). However, some `vsi` files have extra data attached to them, meaning that the index of the full resolution image is not located at the first series. To adapt this you can edit the `full_index` variable in the `loc.py` file and set it t the index used by your vsi file. 

## 4. Plaque Categorisation: Supervised Learning with ResNet50
The categorisation stage handles the classification of located plaques into different types. Running files for this stage with different paramters requires changing YAML files in the `configs` directory.

A ResNet50 model is fine-tuned using labeled data to classify plaques into nine categories (`Diffuse`, `Cored`, `Compact`, `Coarse`, `CAA`, `Subpial`, `OtherAB`, `UndefAB`, `NonAB`), as discussed in the [paper](https://resolver.tudelft.nl/uuid:c1f7047c-43aa-4c9a-955a-396ea1f317ad).

### Fine-Tuning Models

To fine-tune models on different folds of the labeled plaque dataset run:
```
python -u scripts/categorisation/supervised/finetune.py -c configs/supervised.yaml
```

The weights for ResNet50 are taken by default from a pre-trained Momentum Contrast (MoCo) [model](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md). Weights can also be used from pre-trained Simple Contrastive Learning (SimCLR) model by setting ` model_type: simclr` in `supervised.yaml`.

The fine-tuned models trained on different folds are saved as different versions under the `models/categorisation` directory.
### Calculating Class Probability Thresholds

Once the models are fine-tuned, calculate class probability thresholds on the validation sets using `evaluate.py`. Make sure to set `generate_thresholds: True` in the `supervised.yaml` config file before executing the script. Once this is done, run:
```
python -u scripts/categorisation/supervised/evaluate.py -c configs/supervised.yaml
```

###  Evaluating Models

To evaluate the models on their corresponding test sets run `evaluate.py` with `generate_thresholds: False` in the `supervised.yaml` config file.
```
python -u scripts/categorisation/supervised/evaluate.py -c configs/supervised.yaml
```
Logs of the evaluation, along with a confusion matrix, are saved under the `evaluation` directory.

###  Making Predictions

Once the models are trained and their thresholds generated, class predictions can be made for the complete unlabeled dataset by running:
```
python -u scripts/categorisation/supervised/predict.py -c configs/supervised.yaml
```
In case you'd like to make predictions on only the labeled plaques, execute the previous script with `use_labeled_data: True` in `supervised.yaml`.

###  Visualising Predictions
To visualise which classes plaques are assigned to, run:

```
python -u scripts/categorisation/class_report.py -noi <NUM_IMAGES_TO_SAVE_PER_CLASS> -ip <HDF5_IMAGE_FILES_PATH> -ap <SAVED_ASSIGNMENTS_PATH>
```
```
python -u scripts/categorisation/visualisation.py -sup -ipc <NUM_IMAGES_TO_SAVE_PER_CLASS> -is <IMG_SIZE> -ap <SAVED_ASSIGNMENTS_PATH>
```
Plaque assignments should be saved as an image grid under the `result/class_report` directory. For another example on how to run these files, it is recommended to view `pipeline.sh`.

### Plots for Analysing Predictions
To plot class distributions predicted by the model, use the commands below. Plots are saved by default under the `result/stats_plots` directory. View the `plots.yaml` config file for changing parameters.

To plot plaque frequency counts for the different classes predicted by the ensemble model use:
```
python -u scripts/categorisation/stats/freq_bar_plots.py -c configs/plots.yaml
```

#### AB Load Distribution Plots

For plots concerning AB loads for different classes, first compute the loads based on the ensemble model's predictions:
```
python -u scripts/categorisation/compute_load.py -c configs/plots.yaml
```

To generate box plots (overlaid with violin plots) for the AB loads of the different classes, run:
```
python -u scripts/categorisation/stats/load_violin_plots.py -c configs/plots.yaml
```

To generate correlation matrices based on collected data of centenarian participants, use:
```
python -u scripts/categorisation/stats/correlation_matrices.py -c configs/plots.yaml
```

To generate scatter plots of predicted AB loads for CAA vs. Thal CAA and Cored vs. CERAD NP scores, use:
```
python -u scripts/categorisation/stats/phase_scatter_plots.py -c configs/plots.yaml
```

## Annotating Plaques
There is an option for annotating plaques to expand the labeled dataset. The current method relies on generating image grids which consist of randomly chosen plaques from the unlabeled dataset. These generated image grids are saved as samples under the `labeled_plaque_samples/samples` directory. For changing grid parameters, edit the `annotations.yaml` config file.

To generate the image grid samples:
```
python -u scripts/categorisation/annotations/create_samples.py -c configs/annotations.yaml
```
To annotate a new plaque, add it as an entry to the `labeled_plaque_samples/labeled_data.csv` file. Each plaque entry should consist of: number of the image grid sample it originated from; the row in the grid; column in the grid; and the true class label that plaque belongs to. Make sure the label values align with the names indicated in `labeled_plaque_samples/label_names.csv`. Once these entries have been added to the CSV file, collect the label information by running:
```
python -u scripts/categorisation/annotations/collect_labeled_plaques.py -c configs/annotations.yaml
```
References to the plaque images are then saved in NPZ format. If you would like to additionally make a copy of the labeled plaque images, run:
```
python -u scripts/categorisation/annotations/copy_labeled_plaques.py -c configs/annotations.yaml
```


## Plaque Categorisation: Other Methods
Other unsupervised methods for categorising plaques were explored during the Master's thesis. These turned out to not be successful. Instructions on how to run some of these are included below.

### Autoencoders
A Variational Autoencoder (VAE) can be fine-tuned on the unlabeled plaque dataset:
```
python -u scripts/categorisation/autoencoders/finetune.py -c configs/vae.yaml
```

The VAE will learn lower-dimensional representations of the plaque images by trying to reconstruct the input. The trained model is saved under the `models/categorisation` directory. Once the model has been trained, image representations are extracted from the VAE encoder's output for further processing (e. g. unsupervised clustering).

### Unsupervised Clustering
K-Means and OPTICS are available as unsupervised clustering methods. These models are fitted to extracted features and used to cluster plaques into different groups. These features can include the area and roundness of a plaque, pre-trained features from AlexNet or ResNet, or features extracted from the autoencoder.

First compute and save features for each plaque with a specified model:
```
python -u scripts/categorisation/feature_clustering/features.py -f <MODEL_TYPE> -r <HDF5_IMAGE_FILES_PATH> 
```

Once the features are computed, they can be used to fit one of the unsupervised clustering models with the `--fit` parameter. To additionally save predictions for plaques include the `--predict` parameter.

Example for generating predictions with OPTICS using AlexNet and basic plaque features:
```
python -u scripts/categorisation/feature_clustering/cluster.py --alex_features --simple_features -m optics --fit --predict -r <HDF5_IMAGE_FILES_PATH> 
```
Fitted unsupervised clustering models are saved under the `result/cluster_model` directory. OPTICS plots are saved under `result/optics`. If t-SNE dimensionality reduction is enabled with `--tsne`, t-SNE plots are saved under `result/tsne`.

## Code Formatting
For auto-formatting code please run in the root directory:

```
black .
```

For auto-formatting imports please run in the root directory:

```
isort .
```
