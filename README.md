# Amyloid-beta plaque quantification and analysis
This repository houses the code base that accompanies the paper [Amyloid-beta plaque quantification and analysis](https://repository.tudelft.nl/islandora/object/uuid:7b67aaa9-41a5-4bc3-b987-68f34ed145d9?collection=education) by Chiel de Vries. This work was done as the graduation project of the author, who graduated successfully from The TU Delft. The repository contains the code necessary to reproduce the results of the paper, but not the dataset. The dataset contains personally identifiable data, therefore it cannot be shared publicly. 

## Installation
To be able to run the code, several packages should be installed after cloning or forking the project. This can be done in two ways: Anaconda and pip.
> :warning: **Disclaimer**: both methods result in slightly different environments. both methods exist because the anaconda version was necessary for the HPC, but it didn't work on my local machine. The pip version does, but is slightly different as a result. The code was tested on both versions and should run fine.

### Anaconda 
To run the code on the TU Delft HPC, it is best to use anaconda for installation. The `Abeta.yml` file contains all the necessary packages so running `conda env create -f Abeta.yml` should install the environment correctly. From here the code can be run.

### Pip
If you prefer pip, there is also a `requirements.txt` file that can be used to install the packages. After creating your virtual environment, run `pip install -r requirements.txt` to install all packages. 

