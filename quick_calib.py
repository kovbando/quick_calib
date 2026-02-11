import argparse
import concurrent.futures
import logging
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Checkerboard size as inner corners (columns, rows)
CHECKERBOARD_SIZE = (9, 6)
SQUARE_SIZE = 1.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def _list_images(input_dir: Path) -> List[Path]:
    return sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def _detect_checkerboard(
    image_path: Path, checkerboard_size: Tuple[int, int]
) -> Tuple[Path, bool, Optional[np.ndarray], Optional[Tuple[int, int]]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return image_path, False, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)
    if not found:
        return image_path, False, None, None

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return image_path, True, corners.reshape(-1, 2), (gray.shape[1], gray.shape[0])


def _prepare_object_points(
    checkerboard_size: Tuple[int, int], square_size: float
) -> np.ndarray:
    cols, rows = checkerboard_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def _compute_reprojection_error(
    objp: np.ndarray,
    imgp: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    return float(np.linalg.norm(imgp - projected, axis=1).mean())


def _run_detection(
    image_paths: Iterable[Path], checkerboard_size: Tuple[int, int]
) -> Tuple[List[Path], List[np.ndarray], List[Tuple[int, int]]]:
    valid_paths: List[Path] = []
    image_points: List[np.ndarray] = []
    image_sizes: List[Tuple[int, int]] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(_detect_checkerboard, path, checkerboard_size)
            for path in image_paths
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Detecting checkerboards",
        ):
            image_path, found, corners, image_size = future.result()
            if found and corners is not None and image_size is not None:
                valid_paths.append(image_path)
                image_points.append(corners)
                image_sizes.append(image_size)
                logging.info("Found checkerboard in %s", image_path.name)
            else:
                logging.info("No checkerboard in %s", image_path.name)

    return valid_paths, image_points, image_sizes


def _save_matrix(found_dir: Path, camera_matrix: np.ndarray) -> None:
    try:
        from scipy.io import savemat
    except ImportError as exc:
        raise SystemExit(
            "scipy is required to save K.mat. Install with: pip install scipy"
        ) from exc

    output_path = found_dir / "K.mat"
    savemat(str(output_path), {"K": camera_matrix})
    logging.info("Saved calibration matrix to %s", output_path)


def _copy_found_images(found_dir: Path, image_paths: Iterable[Path]) -> None:
    found_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        shutil.copy2(image_path, found_dir / image_path.name)


def _validate_image_sizes(image_sizes: List[Tuple[int, int]]) -> Tuple[int, int]:
    if not image_sizes:
        raise SystemExit("No valid images were detected.")

    reference = image_sizes[0]
    for size in image_sizes[1:]:
        if size != reference:
            raise SystemExit(
                "Images have different sizes. Please provide same-size images."
            )
    return reference


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Camera calibration from checkerboard images."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing images with checkerboards",
    )
    args = parser.parse_args()

    _init_logging()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        logging.error("Input directory does not exist: %s", input_dir)
        return 1

    image_paths = _list_images(input_dir)
    if not image_paths:
        logging.error("No images found in %s", input_dir)
        return 1

    logging.info("Found %d images", len(image_paths))
    valid_paths, image_points, image_sizes = _run_detection(
        image_paths, CHECKERBOARD_SIZE
    )

    if len(valid_paths) < 3:
        logging.error("Need at least 3 valid checkerboard images, found %d", len(valid_paths))
        return 1

    found_dir = input_dir / "found"
    _copy_found_images(found_dir, valid_paths)
    logging.info("Copied %d images to %s", len(valid_paths), found_dir)

    image_size = _validate_image_sizes(image_sizes)
    objp = _prepare_object_points(CHECKERBOARD_SIZE, SQUARE_SIZE)
    object_points = [objp for _ in image_points]

    logging.info("Running camera calibration")
    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    logging.info("Computing reprojection errors")
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(
                _compute_reprojection_error,
                objp,
                imgp,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )
            for imgp, rvec, tvec in zip(image_points, rvecs, tvecs)
        ]
        errors = []
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Calibration progress",
        ):
            errors.append(future.result())

    mean_error = float(np.mean(errors)) if errors else 0.0
    logging.info("Mean reprojection error: %.4f", mean_error)

    _save_matrix(found_dir, camera_matrix)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
