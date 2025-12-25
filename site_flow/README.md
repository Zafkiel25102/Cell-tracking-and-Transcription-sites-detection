# Transcription sites Analysis

This part contains the code for the RNA transcription site analysis pipeline, including site detection, registration, tracking, and intensity computation.

## Environment Setup

### Based on Cell Segmentation Environment (suggested)

The most suggested way to set up the environment is to use the same environment as Cell Segmentation, while with following addtional packages installed:

```shell
conda activate sam-yolo
pip install trackpy SimpleITK
```

### Build from Scratch

This code based on `python==3.9.18`，and requires `pytorch==2.0.0` and `torchvision==0.15.0`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Other package dependencies:

```
pip install tifffile trackpy SimpleITK scikit-image scikit-image
```

## Structure

The directory structure:

```
├── site_flow
│   ├── README.md                  <- This README file
|   ├── predictor.py               <- Main predictor class for site analysis
|   ├── utils.py                   <- Utility functions
|   ├── run.py                     <- Example script to run the site analysis pipeline
|   ├── pt                         <- Pretrained model weights folder
|   ├── example                    <- Example data folder
│   │    ├── cellraw_20486.tif     <- Example cell sequence tiff file
│   │    ├── cellraw_20486         <- Analysis result folder
```

## Data preparation

This pipeline expects the input data to be in the following format:

- A folder containing TIFF files of the cell sequence images.

TIFF files should be named in a way that indicates the cell id, such as `cellraw_0001.tif`, `cellraw_0002.tif`, etc. The images should be grayscale and have a consistent resolution. In this pipeline, we using the fixed resolution of `128*128` pixels for each cell image. 

## Usage

**Note**: We design different tracking methods for cell sequences with different transcription sites.

- For cell sequence which at most has one transcription site, we link sites in consecutive and constrained frames into *patch*, 
    and then link patches into *trajectory* based on the distance between the patches.

- For cell sequence which has 2 transcription sites, we use the cluster method to assign the transcription sites to different clusters.

- **IMPORTANT!!**  All site tracking are done in the registration space, which is the space after the registration of the cell sequence.

```python
import shutil
from os.path import join
from pathlib import Path
import torch

from predictor import SitePredictor

cell_seq_path = Path('path/to/cell_sequence_folder/tiff_file')
cell_seq_data_dir = cell_seq_path.parent / cell_seq_path.stem
cell_seq_data_dir.mkdir(exist_ok=True)
shutil.copy(cell_seq_path, cell_seq_data_dir, follow_symlinks=True)

site_predictor = SitePredictor(
    cell_seq_data_dir, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

model_dir = 'path/to/pt'
# Run the site detection and registration
spotlearn_model_path = join(model_dir,'spotlearn/epoch40.pt')
site_predictor.site_detect(spotlearn_model_path)
site_predictor.registration_recursive()
# Get the coordinates of the transcription site 
rf_classifier_path = join(model_dir,'rf_classifier/random_forest_model.pkl')
nn_classifier_path = join(model_dir,'nn_classifier/tut1-model.pt')
site_predictor.get_mask_coor_reg(
    rf_classifier_path=rf_classifier_path,
    nn_classifier_path=nn_classifier_path,
)

# PAY ATTENTION:
# Choose the appropriate tracking method based on the transcription sites in the cell sequence.
# For cell sequence which at most has one transcription site
site_predictor.site_track()
# For cell sequence which has 2 transcription sites
site_predictor.site_cluster()

# Compute the intensity of the transcription site
have_two_sites = False  # Set to True if the cell sequence has 2 transcription sites
site_predictor.compute_intensity(site2=have_two_sites)
# Plot the raw stack with tracked sites coordinates
site_predictor.get_raw_stack_with_label()
```

### Example

We provide an example script `run.py` in the `site_flow` folder and an example dataset in the `example` folder to demonstrate the usage of the transcription site analysis pipeline.

```bash
cd site_flow
conda activate sam-yolo
python run.py
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
