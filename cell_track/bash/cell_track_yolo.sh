#!/bin/bash
#SBATCH -p v100
#SBATCH -q gpu
#SBATCH -c 4
#SBATCH --mem 140G
#SBATCH --gres=gpu:1
gpu_='0'         #gpu_='1,2,3'

root_path='/mnt/sda/cell_data/l3/20240711/SOX2_wt_60h_20240717_101207/'    #change rootpath

conda activate cell-track
python Creatdatafloder.py $gpu_ $root_path   


directories=$(find "$root_path" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | sort)

for dir in $directories; do
    echo "Found directory: $dir"
    file_name="$dir"
    echo "SAMPRE_HIGH_INTENSITY"
    python SAMPRE_HIGH_INTENSITY.py $gpu_ $root_path $file_name
    echo "SAMPRE_HIGH_INTENSITY_done"
    # change environment
    echo "sam_yolo_pipeline"
    conda deactivate
    conda activate sam
    python sam_yolo_pipeline.py $gpu_ $root_path $file_name
    echo "sam_yolo_pipeline"
    # change environment
    conda deactivate
    conda activate cell-track
    echo "SEGpostprocess"
    python SEGpostprocess.py $gpu_ $root_path $file_name
    echo "SEGpostprocess_done"
    python rename_copy.py $gpu_ $root_path $file_name
    echo "rename_copy_done"
    echo "preprocess_seq2graph_clean"
    python ./cell-tracker-gnn-main/preprocess_seq2graph_clean.py $gpu_ $root_path $file_name
    echo "preprocess_seq2graph_clean_done"
    echo "inference_clean"
    python ./cell-tracker-gnn-main/inference_clean.py $gpu_ $root_path $file_name
    echo "inference_clean_done"
    echo "postprocess_clean"
    python postprocess_clean.py $gpu_ $root_path $file_name
    echo "postprocess_clean_done"
    python single_fast.py $gpu_ $root_path $file_name
done
