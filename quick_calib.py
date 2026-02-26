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
CHECKERBOARD_SIZE = (8, 6)
SQUARE_SIZE = 1.0
# Camera pixel size in micrometers. Set to 0 to disable mm focal length output.
#PIXEL_SIZE_UM = 4.8 # Use this for typical 1/2.3" sensors. Adjust as needed for your camera.
PIXEL_SIZE_UM = 3.45 # Use this for typical 1/3" sensors. Adjust as needed for your camera.
# Max number of images to use for calibration. Set to 0 to use all.
MAX_CALIB_IMAGES = 100

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


def _save_matrix_text(found_dir: Path, camera_matrix: np.ndarray) -> None:
    output_path = found_dir / "K.txt"
    np.savetxt(str(output_path), camera_matrix, fmt="%.10f")
    logging.info("Saved calibration matrix text to %s", output_path)


def _save_summary(found_dir: Path, summary_lines: List[str]) -> None:
    output_path = found_dir / "summary.txt"
    output_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    logging.info("Saved calibration summary to %s", output_path)


def _compute_fov_summary(
    camera_matrix: np.ndarray,
    image_size: Tuple[int, int],
    total_images: int,
    valid_images: int,
    used_images: int,
    mean_error: float,
    rvecs: Optional[List[np.ndarray]] = None,
    tvecs: Optional[List[np.ndarray]] = None,
) -> Tuple[List[str], List[str]]:
    width, height = image_size
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    favg = 0.5 * (fx + fy)

    fov_x = 2.0 * np.degrees(np.arctan(width / (2.0 * fx)))
    fov_y = 2.0 * np.degrees(np.arctan(height / (2.0 * fy)))
    diag = float(np.hypot(width, height))
    fdiag = 0.5 * (fx + fy)
    fov_d = 2.0 * np.degrees(np.arctan(diag / (2.0 * fdiag)))

    focal_mm_line = None
    if PIXEL_SIZE_UM > 0.0:
        pixel_size_mm = PIXEL_SIZE_UM / 1000.0
        fx_mm = fx * pixel_size_mm
        fy_mm = fy * pixel_size_mm
        favg_mm = favg * pixel_size_mm
        focal_mm_line = (
            f"Focal length (mm): fx={fx_mm:.4f}, fy={fy_mm:.4f}, avg={favg_mm:.4f}"
        )

    summary_lines = [
        "Calibration summary:",
        f"Images in input dir: {total_images}",
        f"Images with checkerboard: {valid_images}",
        f"Images used for calibration: {used_images}",
        f"Mean reprojection error: {mean_error:.4f}",
        f"Image resolution: {width} x {height}",
        f"Focal length (px): fx={fx:.4f}, fy={fy:.4f}, avg={favg:.4f}",
        f"Horizontal FOV (deg): {fov_x:.4f}",
        f"Vertical FOV (deg): {fov_y:.4f}",
        f"Diagonal FOV (deg): {fov_d:.4f}",
        f"Principal point (px): cx={cx:.2f}, cy={cy:.2f}",
        "",
        "Camera matrix K:",
        np.array2string(camera_matrix, precision=6, max_line_width=120),
    ]

    if focal_mm_line is not None:
        summary_lines.insert(6, focal_mm_line)

    # Add rvecs and tvecs if provided
    if rvecs is not None and tvecs is not None:
        summary_lines.append("")
        summary_lines.append("Rotation vectors (rvecs):")
        for i, rvec in enumerate(rvecs):
            summary_lines.append(f"  Image {i}: {rvec.flatten().tolist()}")
        summary_lines.append("")
        summary_lines.append("Translation vectors (tvecs):")
        for i, tvec in enumerate(tvecs):
            summary_lines.append(f"  Image {i}: {tvec.flatten().tolist()}")

    log_lines = [
        f"Focal length (px): fx={fx:.4f}, fy={fy:.4f}, avg={favg:.4f}",
        f"Horizontal FOV (deg): {fov_x:.4f}",
        f"Vertical FOV (deg): {fov_y:.4f}",
        f"Diagonal FOV (deg): {fov_d:.4f}",
        f"Principal point (px): cx={cx:.2f}, cy={cy:.2f}",
    ]

    if focal_mm_line is not None:
        log_lines.insert(1, focal_mm_line)

    return summary_lines, log_lines


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


def _select_calibration_subset(
    valid_paths: List[Path],
    image_points: List[np.ndarray],
    image_sizes: List[Tuple[int, int]],
    max_images: int,
) -> Tuple[List[Path], List[np.ndarray], List[Tuple[int, int]]]:
    if max_images <= 0 or len(valid_paths) <= max_images:
        return valid_paths, image_points, image_sizes

    indices = np.linspace(0, len(valid_paths) - 1, max_images, dtype=int)
    selected_paths = [valid_paths[i] for i in indices]
    selected_points = [image_points[i] for i in indices]
    selected_sizes = [image_sizes[i] for i in indices]
    return selected_paths, selected_points, selected_sizes


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
    total_valid_images = len(valid_paths)

    if MAX_CALIB_IMAGES > 0 and len(valid_paths) > MAX_CALIB_IMAGES:
        skip_count = len(valid_paths) - MAX_CALIB_IMAGES
        logging.info(
            "Limiting calibration to %d images (skipping %d)",
            MAX_CALIB_IMAGES,
            skip_count,
        )
    valid_paths, image_points, image_sizes = _select_calibration_subset(
        valid_paths,
        image_points,
        image_sizes,
        MAX_CALIB_IMAGES,
    )

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

    logging.info("Camera matrix K:\n%s", np.array2string(camera_matrix, precision=6))

    summary_lines, log_lines = _compute_fov_summary(
        camera_matrix,
        image_size,
        total_images=len(image_paths),
        valid_images=total_valid_images,
        used_images=len(image_points),
        mean_error=mean_error,
        rvecs=rvecs,
        tvecs=tvecs,
    )
    for line in log_lines:
        logging.info(line)

    _save_matrix(found_dir, camera_matrix)
    _save_matrix_text(found_dir, camera_matrix)
    _save_summary(found_dir, summary_lines)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
