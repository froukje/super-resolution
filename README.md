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
* added mlflow

## Training 
### SRResNet
* run script with desired parameters, e.g. ```python3 srresnet.py --n-blocks 8 --batch-size 12 --n-channels 22 --gpus 1 --n-epochs 200```
* use batch script saved in ```jobs/start_job.sh``` or ```start_job_nni.sh``` to include hyperparameter tuning using ```nni```
* results are not reported to nni, but to mlflow
* example files for hyperparamter tuning are in ```nni/config/config.py``` and ```nni/search_space/search_space.json```

## Evaluation 
* example plots are in notebooks/plot-predictions.ipynb
![example prediction](notebooks/example_prediction.png)
