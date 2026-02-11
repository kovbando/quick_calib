# quick_calib
- Camera calibration tool for generating a 3x3 intrinsic matrix from checkerboard images.
- For **way better** results use a calibration object with a white bezel around the checkerboard pattern

## Overview
1. Scans an input directory for images.
2. Detects checkerboards in parallel and copies valid images into a found folder.
3. Runs OpenCV calibration to estimate the intrinsic matrix K.
4. Computes reprojection error in parallel for reporting.
5. Saves K as K.mat and K.txt and prints it to the console.

## Requirements
- Python 3.8+
- See requirements.txt for dependencies
- `pip install -r requirements.txt`

## Configuration
Edit the checkerboard size at the top of quick_calib.py:
CHECKERBOARD_SIZE = (cols, rows)

## Usage
`python quick_calib.py path\to\images`

## Output
- found\ (subfolder with detected images)
- found\K.mat (MAT-file with variable K)
- found\K.txt (plain text 3x3 matrix)

## Notes
- For **way better** results use a calibration object with a white bezel around the checkerboard pattern
- All images must have the same resolution.
- At least 3 valid checkerboard images are required.
