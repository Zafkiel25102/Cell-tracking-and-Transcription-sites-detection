import warnings
import shutil
from os.path import join
from pathlib import Path

import tifffile as tiff
import torch

from predictor import SitePredictor
from utils import *

OVERWRITE = True  # Set to True to overwrite existing files

def site_inference(
    cell_seq_path: Path,
    has_two_sites: bool = None,
    model_dir: str = './pt',
    overwrite: bool = OVERWRITE,
    run_det: bool = True,
):
    """Transcription site analysis for a single cell sequence.
    
    Args:
        cell_seq_path (Path): Path to the cell sequence image.
        has_two_sites (bool): Whether the cell sequence has two transcription sites.
        model_dir (str): Directory containing the pre-trained models.
        overwrite (bool): Whether to overwrite existing files.
        run_det (bool): Whether to run the detection and registration steps.
    """

    # Create data directory for the cell sequence
    cell_seq_data_dir = cell_seq_path.parent / cell_seq_path.stem
    cell_seq_data_dir.mkdir(exist_ok=True)
    shutil.copy(cell_seq_path, cell_seq_data_dir, follow_symlinks=True)

    cell_seq = tiff.imread(cell_seq_path)
    if cell_seq.ndim == 2 or cell_seq.shape[0] == 1:
        warnings.warn(f"Cell sequence {cell_seq_path} is a single frame image, skipping further processing.")
        return   #TODO: handle single frame image

    # Create processing object
    site_predictor = SitePredictor(
        cell_seq_data_dir, 
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    if (not (cell_seq_data_dir / 'imgs_raw_mask_reg_rcs.tif').exists()) or overwrite:
        if run_det:
            # Run the site detection and registration
            site_predictor.site_detect(
                model_path=join(model_dir,'spotlearn/epoch40.pt')
            )
        site_predictor.site_reg()
        # Get the coordinates of the transcription site 
        site_predictor.get_mask_coor_reg(
            rf_classifier_path=join(model_dir,'rf_classifier/random_forest_model.pkl'),
            nn_classifier_path=join(model_dir,'nn_classifier/tut1-model.pt'),
        )
    
    if (not (cell_seq_data_dir / 'trajectories_data.csv').exists()) or overwrite:
        # PAY ATTENTION:
        # Choose the appropriate tracking method based on the transcription sites in the cell sequence.
        if not has_two_sites:
            # For cell sequence which at most has one transcription site 
            site_predictor.site_track(
                search_range = 9, 
                memory= 5, 
                threshold = 2, 
            )
        else:
            # For cell sequence which has 2 transcription sites
            site_predictor.site_cluster()
    
    if (not ((cell_seq_data_dir / 'raw_stack_with_label.tif').exists() \
                or (cell_seq_data_dir / 'dataAnalysis_tj_empty_withBg.csv').exists())) \
            or overwrite:
        # Compute the intensity of the transcription site
        site_predictor.compute_intensity(site2=has_two_sites)
    
    del site_predictor


if __name__ == '__main__':

    cell_seq_dir = Path("./example/cellraw_20486.tif")
    site_inference(
        cell_seq_path=cell_seq_dir,
        has_two_sites=False,
    )