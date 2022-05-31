# Super Resolution

## Data
* data from [kaggle.com](https://www.kaggle.com/datasets/akhileshdkapse/super-image-resolution)
* raw data saved in data/HR (high resolution) and data/LR (low resolution)
* 100 images
* data exploration: notebooks/data-exploration.ipynb 

## Preprocessing
* preprocessing.py
    * splits the data in validation and training
    * saves splits in h5 files to data/train.h5 and data/valid.h5

## Model
* train 2 models:
    * SRResNet
    * SRGAN (ToDo)
* based on: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, Ledig et al. (2017), https://doi.org/10.48550/arXiv.1609.04802
* model code based on https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
* converted to pytorch lightning
* added NNI for hyperparamtertuning

## Training (ToDo)

## Evaluation (ToDo) 
