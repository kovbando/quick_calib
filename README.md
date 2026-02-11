# quick_calib
- Camera calibration tool for generating a 3x3 intrinsic matrix from checkerboard images.
- For **way better** results use a calibration object with a white bezel around the checkerboard pattern

## Overview
1. Scans an input directory for images.
2. Detects checkerboards in parallel and copies valid images into a found folder.
3. Runs OpenCV calibration to estimate the intrinsic matrix K.
4. Computes reprojection error in parallel for reporting.
5. Saves K as K.mat and K.txt, prints it to the console, and writes a summary.

## Requirements
- Python 3.8+ (3.14 recommended)
- See requirements.txt for dependencies
- `pip install -r requirements.txt`

## Configuration
Edit the following lines at the top of `quick_calib.py` to configure it:
- Checkerboard size: `CHECKERBOARD_SIZE = (cols, rows)`
- Optional pixel size in micrometers : `PIXEL_SIZE_UM = 4.8`
    - If `PIXEL_SIZE_UM` is set to zero, then the focal length will NOT be calculated in milimeters, andd the line will be omitted from the summary.
- Because `cv2.calibrateCamera` is really slow on large number of input images, and the runtime scales almost exponentially, there is a configurable variable to limit the maximum number of images to be used in calibration.
    - Set `MAX_CALIB_IMAGES = 100` to your desired amount of pictures. Set it to 0 to use all available pictures.
    - **be careful, this can quickly lead to unreasonably long runtimes**

## Usage
after installing the required packages, check the top of the script for the configurable constants, then run:\
`python quick_calib.py path\to\images`

## Output
- found\ (subfolder with detected images)
- found\K.mat (MAT-file with variable K)
- found\K.txt (plain text 3x3 matrix)
- found\summary.txt (FOVs, principal point, focal length, and matrix copy)

## Notes
- For **way better** results use a calibration object with a white bezel around the checkerboard pattern
- It is fully normal for the script to hang, appear to be frozen when running the calibration itself. When the log shows `[INFO] Running camera calibration` it will do its thing, only on a single CPU core, for a long time.
- All images must have the same resolution.
- At least 3 valid checkerboard images are required.
- Summary includes horizontal/vertical/diagonal FOV and principal point, and focal length in mm if PIXEL_SIZE_UM > 0.
