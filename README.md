# U-Net Vocal Separator

This's a U-Net based tool to extract vocal from songs with Keras framework.

## Installation

### Install the packages

```
cd unet-separator
pip install -r requirements.txt
```
### GPU calculation

Keras supports NVidia GPUs. Is required to install `CUDA Toolkit` and `cuDNN` library.

## Separate vocal
Results are saved as `*_vocal.wav` in `predict_results` folder.

### Run a script
```
python predict.py
```
There are arguments implemented to console calls:

Argument | Description | Default
------------ | ------------- | -------------
--input | path to input file| files/mixture.wav
--output | path to output folder| predict_results/
--hard_mask | threshold for mask, if equals 0 mask is not used [0,1]| 0.0
--loop | amount of loops in predict func (recursive method) | 1
--do_plot | creating a plot of spectrogram | True if used
--gpu | run with GPU | True if used

Argument can be combined and used ind different posiontions.
```
python predict.py --gpu --output path --loop 2
```

## Train model

Database used in this project is [MUSDB18HQ](https://sigsep.github.io/datasets/musdb.html#sisec-2018-evaluation-campaign).

### How dataset looks like
```
musdb18hq/
  +- songs_name_1/
  |    +- mixture.wav
  |    +- vocals.wav
  |    +- ...
  +- songs_name_2/
  |    +- mixture.wav
  |    +- vocals.wav
  ...  +- ...
```
### Preprocess database
Before training is required to preprocess database. Sound files will be saved in spectrogram forms as uncompressed .npz files.

```
python preprocess_data.py
```
Argument | Description | Default
------------ | ------------- | -------------
--input | path to input database |  musdb18hq
--output | path to output database | musdb18hq_npz

### Train a model
```
python train.py
```
Additional arguments:

Argument | Description | Default
------------ | ------------- | -------------
--input | path to input database |  musdb18hq_npz
--output | path to folder for saved weights | train_results
--val_split | proportion of the data to train on | 0.1
--batch | batch size for training | 32
--epochs | number of epochs to train | 100
--gpu | run with GPU | True if used