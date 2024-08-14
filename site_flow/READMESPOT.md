# Transcription sites Analysis

## Usage

This code based on `python`，and requires `pytorch` and `torchvision`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Other package dependencies:

```
pip install tifffile trackpy SimpleITK scikit-image
```

### Data preparation

We expect the directory structure to be the following:

```
data
├── field1                    <- field data folder
│   ├── 0                     <- site num
│   │    ├── cellraw_xxx.tif  <- single cell sequence file
│   ├── 1                     <- site num
│   │    ├── cellraw_xxx.tif  <- single cell sequence file
...
├── field2
...
```

### Inference

Making sure your data and python environment is fine, then run:

```
python predict.py --data_path your_data_path --model_weight your_model_weight
```

## Citation

```
Gudla et. al., "SpotLearn: Convolutional Neural Network for Detection of Fluorescence In Situ Hybridization (FISH) Signals in
High-Throughput Imaging Approaches". Cold Spring Harb Symp Quant Biol. 2017 Nov 28. pii: 033761. doi: 10.1101/sqb.2017.82.033761.

R. Beare, B. C. Lowekamp, Z. Yaniv, "Image Segmentation, Registration and Characterization in R with SimpleITK", J Stat Software
, 86(8), https://doi.org/10.18637/jss.v086.i08, 2018.

Allan, D. B., Caswell, T., Keim, N. C., van der Wel, C. M., & Verweij, R. W. (2023). soft-matter/trackpy: v0.6.1 (v0.6.1).
Zenodo. https://doi.org/10.5281/zenodo.7670439
```
