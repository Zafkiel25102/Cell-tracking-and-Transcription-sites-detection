#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import sys

def main():
    current_dir = sys.argv[1]  # Assuming the directory is passed as the first argument

    # List all .tif files in the current directory
    tif_files = [file for file in os.listdir(current_dir) if file.endswith(".tif")]

    for file in tif_files:
        file_name = os.path.splitext(file)[0]

        # Create a directory with the name of the .tif file
        new_dir = os.path.join(current_dir, file_name)
        os.makedirs(new_dir, exist_ok=True)

        # Create subdirectories
        sub_dir_01 = os.path.join(new_dir, "01")
        os.makedirs(sub_dir_01, exist_ok=True)

        sub_dir_01_gt = os.path.join(new_dir, "01_GT")
        os.makedirs(sub_dir_01_gt, exist_ok=True)

        # Move the original .tif file to the 01 subdirectory
        src_file = os.path.join(current_dir, file)
        dst_file = os.path.join(sub_dir_01, file)
        shutil.move(src_file, dst_file)

if __name__ == "__main__":
    main()