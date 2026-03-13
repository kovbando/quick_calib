import argparse
import concurrent.futures
import logging
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Checkerboard size as inner corners (columns, rows)
CHECKERBOARD_SIZE = (8, 11)
SQUARE_SIZE = 1.0
# Set to True for checkerboards decorated with ChArUco markers.
# When enabled, ArUco/ChArUco detections are used to stabilize checkerboard orientation.
HAS_CHARUCO_MARKERS = True
# OpenCV ArUco dictionary name. Examples: DICT_4X4_50, DICT_5X5_1000.
CHARUCO_ARUCO_DICT = "DICT_5X5_1000"
# Number of checker squares in X/Y for ChArUco board creation.
# For a checkerboard with inner corners (cols, rows), these are typically (cols + 1, rows + 1).
CHARUCO_SQUARES_X = CHECKERBOARD_SIZE[0] + 1
CHARUCO_SQUARES_Y = CHECKERBOARD_SIZE[1] + 1
# Marker side length in the same unit as SQUARE_SIZE.
CHARUCO_MARKER_LENGTH = SQUARE_SIZE * 0.7
# Camera pixel size in micrometers. Set to 0 to disable mm focal length output.
#PIXEL_SIZE_UM = 4.8 # Use this for typical 1/2.3" sensors. Adjust as needed for your camera.
#PIXEL_SIZE_UM = 3.45 # Use this for typical 1/3" sensors. Adjust as needed for your camera.
PIXEL_SIZE_UM = 0.0 # Set to 0 to disable mm focal length output.
# Max number of images to use for calibration. Set to 0 to use all.
MAX_CALIB_IMAGES = 100

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _get_aruco_dictionary(dict_name: str):
    aruco = cv2.aruco
    dict_id = getattr(aruco, dict_name, None)
    if dict_id is None:
        raise ValueError(
            f"Unknown ArUco dictionary '{dict_name}'. "
            "Use a valid OpenCV dictionary constant name, e.g. DICT_4X4_50."
        )
    return aruco.getPredefinedDictionary(dict_id)


def _create_charuco_board(cols: int, rows: int, square_length: float, marker_length: float, dictionary):
    aruco = cv2.aruco
    if hasattr(aruco, "CharucoBoard"):
        return aruco.CharucoBoard((cols, rows), square_length, marker_length, dictionary)
    return aruco.CharucoBoard_create(cols, rows, square_length, marker_length, dictionary)


def _generate_orientation_permutations(cols: int, rows: int) -> List[np.ndarray]:
    idx = np.arange(rows * cols).reshape(rows, cols)
    permutations = [
        idx.copy(),
        np.fliplr(idx),
        np.flipud(idx),
        np.flipud(np.fliplr(idx)),
    ]
    return [p.reshape(-1) for p in permutations]


def _detect_charuco_observations(
    gray: np.ndarray,
) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    aruco = cv2.aruco
    dictionary = _get_aruco_dictionary(CHARUCO_ARUCO_DICT)
    params = aruco.DetectorParameters()
    marker_corners, marker_ids, _ = aruco.detectMarkers(
        gray,
        dictionary,
        parameters=params,
    )
    if marker_ids is None or len(marker_ids) == 0:
        return marker_corners, marker_ids, None, None

    board = _create_charuco_board(
        CHARUCO_SQUARES_X,
        CHARUCO_SQUARES_Y,
        SQUARE_SIZE,
        CHARUCO_MARKER_LENGTH,
        dictionary,
    )
    _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
    )
    return marker_corners, marker_ids, charuco_corners, charuco_ids


def _reorder_corners_with_charuco(
    checker_corners: np.ndarray,
    checkerboard_size: Tuple[int, int],
    marker_corners: Optional[List[np.ndarray]],
    marker_ids: Optional[np.ndarray],
    charuco_corners: Optional[np.ndarray],
    charuco_ids: Optional[np.ndarray],
) -> Tuple[np.ndarray, bool, bool, int]:
    cols, rows = checkerboard_size
    if checker_corners.shape[0] != cols * rows:
        return checker_corners, False, False, 0

    if marker_ids is None or marker_corners is None or len(marker_ids) < 3:
        return checker_corners, False, False, 0

    board = _create_charuco_board(
        CHARUCO_SQUARES_X,
        CHARUCO_SQUARES_Y,
        SQUARE_SIZE,
        CHARUCO_MARKER_LENGTH,
        _get_aruco_dictionary(CHARUCO_ARUCO_DICT),
    )

    if hasattr(board, "getChessboardCorners"):
        board_corners_3d = board.getChessboardCorners()
    else:
        board_corners_3d = board.chessboardCorners
    board_corners_2d = np.asarray(board_corners_3d, dtype=np.float32).reshape(-1, 3)[:, :2]
    if board_corners_2d.shape[0] != cols * rows:
        return checker_corners, False, False, 0

    if hasattr(board, "getIds"):
        board_marker_ids = np.asarray(board.getIds()).reshape(-1)
    else:
        board_marker_ids = np.asarray(board.ids).reshape(-1)
    if hasattr(board, "getObjPoints"):
        board_marker_obj_points = board.getObjPoints()
    else:
        board_marker_obj_points = board.objPoints

    marker_center_by_id: Dict[int, np.ndarray] = {}
    for bid, obj in zip(board_marker_ids, board_marker_obj_points):
        obj = np.asarray(obj, dtype=np.float32).reshape(-1, 3)
        marker_center_by_id[int(bid)] = np.mean(obj[:, :2], axis=0)

    detected_marker_centers: List[Tuple[int, np.ndarray]] = []
    for detected_id, detected_corners in zip(marker_ids.flatten(), marker_corners):
        det_id = int(detected_id)
        if det_id in marker_center_by_id:
            det_corners_2d = np.asarray(detected_corners, dtype=np.float32).reshape(-1, 2)
            detected_marker_centers.append((det_id, np.mean(det_corners_2d, axis=0)))

    if len(detected_marker_centers) < 3:
        return checker_corners, False, False, len(detected_marker_centers)

    expected_count = cols * rows
    valid_pairs: List[Tuple[int, np.ndarray]] = []
    if charuco_ids is not None and charuco_corners is not None:
        for char_id, corner in zip(charuco_ids.flatten(), charuco_corners.reshape(-1, 2)):
            if 0 <= int(char_id) < expected_count:
                valid_pairs.append((int(char_id), corner))

    matched_observations = max(len(detected_marker_centers), len(valid_pairs))

    permutations = _generate_orientation_permutations(cols, rows)
    best_score = float("inf")
    best_perm = permutations[0]
    best_index = 0

    for idx, perm in enumerate(permutations):
        permuted = checker_corners[perm]
        homography, _ = cv2.findHomography(board_corners_2d, permuted.astype(np.float32), 0)
        if homography is None:
            continue

        residuals: List[float] = []

        for marker_id, marker_center_img in detected_marker_centers:
            board_center = marker_center_by_id[marker_id].reshape(1, 1, 2)
            projected = cv2.perspectiveTransform(board_center.astype(np.float32), homography)
            projected_pt = projected.reshape(2)
            residuals.append(float(np.linalg.norm(projected_pt - marker_center_img)))

        for expected_idx, char_pt in valid_pairs:
            residuals.append(float(np.linalg.norm(permuted[expected_idx] - char_pt)))

        score = float(np.mean(residuals)) if residuals else float("inf")
        if score < best_score:
            best_score = score
            best_perm = perm
            best_index = idx

    reordered = checker_corners[best_perm]
    orientation_corrected = best_index != 0
    return reordered, True, orientation_corrected, matched_observations


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
) -> Tuple[
    Path,
    bool,
    Optional[np.ndarray],
    Optional[Tuple[int, int]],
    Optional[Dict[str, Any]],
]:
    image = cv2.imread(str(image_path))
    if image is None:
        return image_path, False, None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)
    if not found:
        return image_path, False, None, None, None

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    corners_2d = corners.reshape(-1, 2)
    charuco_debug: Optional[Dict[str, Any]] = None

    if HAS_CHARUCO_MARKERS:
        try:
            marker_corners, marker_ids, charuco_corners, charuco_ids = _detect_charuco_observations(gray)
            corners_2d, charuco_used, orientation_corrected, matched_pairs = _reorder_corners_with_charuco(
                corners_2d,
                checkerboard_size,
                marker_corners,
                marker_ids,
                charuco_corners,
                charuco_ids,
            )
            charuco_debug = {
                "marker_corners": marker_corners,
                "marker_ids": marker_ids,
                "charuco_corners": charuco_corners,
                "charuco_ids": charuco_ids,
                "charuco_used": charuco_used,
                "orientation_corrected": orientation_corrected,
                "matched_pairs": matched_pairs,
            }
        except (AttributeError, cv2.error, ValueError) as exc:
            logging.warning(
                "ChArUco orientation step failed for %s. Using plain checkerboard ordering. Reason: %s",
                image_path.name,
                exc,
            )
            charuco_debug = {
                "marker_corners": None,
                "marker_ids": None,
                "charuco_corners": None,
                "charuco_ids": None,
                "charuco_used": False,
                "orientation_corrected": False,
                "matched_pairs": 0,
            }

    return image_path, True, corners_2d, (gray.shape[1], gray.shape[0]), charuco_debug


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
) -> Tuple[List[Path], List[np.ndarray], List[Tuple[int, int]], List[Optional[Dict[str, Any]]]]:
    valid_paths: List[Path] = []
    image_points: List[np.ndarray] = []
    image_sizes: List[Tuple[int, int]] = []
    charuco_debug_infos: List[Optional[Dict[str, Any]]] = []

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
            image_path, found, corners, image_size, charuco_debug = future.result()
            if found and corners is not None and image_size is not None:
                valid_paths.append(image_path)
                image_points.append(corners)
                image_sizes.append(image_size)
                charuco_debug_infos.append(charuco_debug)
                logging.info("Found checkerboard in %s", image_path.name)
                if HAS_CHARUCO_MARKERS and charuco_debug is not None:
                    logging.info(
                        "ChArUco status %s: markers=%d, matched_pairs=%d, used=%s, corrected=%s",
                        image_path.name,
                        int(len(charuco_debug.get("marker_ids"))) if charuco_debug.get("marker_ids") is not None else 0,
                        int(charuco_debug.get("matched_pairs", 0)),
                        bool(charuco_debug.get("charuco_used", False)),
                        bool(charuco_debug.get("orientation_corrected", False)),
                    )
            else:
                logging.info("No checkerboard in %s", image_path.name)

    return valid_paths, image_points, image_sizes, charuco_debug_infos


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


def _build_vector_lines(
    vectors: List[np.ndarray],
    calibration_paths: Optional[List[Path]] = None,
) -> List[str]:
    lines: List[str] = []
    for i, vec in enumerate(vectors):
        image_label = f"Image {i}"
        if calibration_paths is not None and i < len(calibration_paths):
            image_label = calibration_paths[i].name
        lines.append(f"{image_label}: {vec.flatten().tolist()}")
    return lines


def _save_vector_file(found_dir: Path, filename: str, lines: List[str]) -> None:
    output_path = found_dir / filename
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Saved vector list to %s", output_path)


def _compute_fov_summary(
    camera_matrix: np.ndarray,
    image_size: Tuple[int, int],
    total_images: int,
    valid_images: int,
    used_images: int,
    mean_error: float,
    rvecs: Optional[List[np.ndarray]] = None,
    tvecs: Optional[List[np.ndarray]] = None,
    calibration_paths: Optional[List[Path]] = None,
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
            image_label = f"Image {i}"
            if calibration_paths is not None and i < len(calibration_paths):
                image_label = calibration_paths[i].name
            summary_lines.append(f"  {image_label}: {rvec.flatten().tolist()}")
        summary_lines.append("")
        summary_lines.append("Translation vectors (tvecs):")
        for i, tvec in enumerate(tvecs):
            image_label = f"Image {i}"
            if calibration_paths is not None and i < len(calibration_paths):
                image_label = calibration_paths[i].name
            summary_lines.append(f"  {image_label}: {tvec.flatten().tolist()}")

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


def _copy_found_images(
    found_dir: Path,
    image_paths: Iterable[Path],
    image_points: Iterable[np.ndarray],
    checkerboard_size: Tuple[int, int],
    charuco_debug_infos: Optional[Iterable[Optional[Dict[str, Any]]]] = None,
) -> None:
    image_paths = list(image_paths)
    image_points = list(image_points)
    found_dir.mkdir(parents=True, exist_ok=True)
    if charuco_debug_infos is None:
        charuco_debug_infos = [None] * len(image_paths)
    else:
        charuco_debug_infos = list(charuco_debug_infos)

    for image_path, corners, charuco_debug in zip(image_paths, image_points, charuco_debug_infos):
        image = cv2.imread(str(image_path))
        output_path = found_dir / image_path.name

        if image is None:
            logging.warning(
                "Could not load %s for annotation. Copying original image.",
                image_path.name,
            )
            shutil.copy2(image_path, output_path)
            continue

        # drawChessboardCorners expects Nx1x2 corner shape.
        corners_for_draw = corners.reshape(-1, 1, 2).astype(np.float32)
        cv2.drawChessboardCorners(image, checkerboard_size, corners_for_draw, True)

        if HAS_CHARUCO_MARKERS and charuco_debug is not None:
            marker_corners = charuco_debug.get("marker_corners")
            marker_ids = charuco_debug.get("marker_ids")
            charuco_corners = charuco_debug.get("charuco_corners")
            charuco_ids = charuco_debug.get("charuco_ids")
            charuco_used = bool(charuco_debug.get("charuco_used", False))
            orientation_corrected = bool(charuco_debug.get("orientation_corrected", False))
            matched_pairs = int(charuco_debug.get("matched_pairs", 0))

            if marker_corners is not None and marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)

            if charuco_corners is not None and charuco_ids is not None:
                for cid, cpt in zip(charuco_ids.flatten(), charuco_corners.reshape(-1, 2)):
                    x = int(round(float(cpt[0])))
                    y = int(round(float(cpt[1])))
                    cv2.circle(image, (x, y), 4, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(
                        image,
                        str(int(cid)),
                        (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

            marker_count = int(len(marker_ids)) if marker_ids is not None else 0
            status_lines = [
                f"ArUco markers: {marker_count}",
                f"ChArUco pairs used: {matched_pairs}",
                f"Used for orientation: {'YES' if charuco_used else 'NO'}",
                f"Orientation corrected: {'YES' if orientation_corrected else 'NO'}",
            ]
            y0 = 24
            for line in status_lines:
                cv2.putText(
                    image,
                    line,
                    (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (20, 220, 20),
                    2,
                    cv2.LINE_AA,
                )
                y0 += 24

        cv2.imwrite(str(output_path), image)


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
    charuco_debug_infos: List[Optional[Dict[str, Any]]],
    max_images: int,
) -> Tuple[List[Path], List[np.ndarray], List[Tuple[int, int]], List[Optional[Dict[str, Any]]]]:
    if max_images <= 0 or len(valid_paths) <= max_images:
        return valid_paths, image_points, image_sizes, charuco_debug_infos

    indices = np.linspace(0, len(valid_paths) - 1, max_images, dtype=int)
    selected_paths = [valid_paths[i] for i in indices]
    selected_points = [image_points[i] for i in indices]
    selected_sizes = [image_sizes[i] for i in indices]
    selected_charuco_infos = [charuco_debug_infos[i] for i in indices]
    return selected_paths, selected_points, selected_sizes, selected_charuco_infos


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
    valid_paths, image_points, image_sizes, charuco_debug_infos = _run_detection(
        image_paths, CHECKERBOARD_SIZE
    )

    if len(valid_paths) < 3:
        logging.error("Need at least 3 valid checkerboard images, found %d", len(valid_paths))
        return 1

    found_dir = input_dir / "found"
    _copy_found_images(
        found_dir,
        valid_paths,
        image_points,
        CHECKERBOARD_SIZE,
        charuco_debug_infos,
    )
    logging.info("Copied %d images to %s", len(valid_paths), found_dir)
    total_valid_images = len(valid_paths)

    if MAX_CALIB_IMAGES > 0 and len(valid_paths) > MAX_CALIB_IMAGES:
        skip_count = len(valid_paths) - MAX_CALIB_IMAGES
        logging.info(
            "Limiting calibration to %d images (skipping %d)",
            MAX_CALIB_IMAGES,
            skip_count,
        )
    valid_paths, image_points, image_sizes, charuco_debug_infos = _select_calibration_subset(
        valid_paths,
        image_points,
        image_sizes,
        charuco_debug_infos,
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
        calibration_paths=valid_paths,
    )
    for line in log_lines:
        logging.info(line)

    _save_matrix(found_dir, camera_matrix)
    _save_matrix_text(found_dir, camera_matrix)
    _save_summary(found_dir, summary_lines)
    r_lines = _build_vector_lines(rvecs, valid_paths)
    t_lines = _build_vector_lines(tvecs, valid_paths)
    _save_vector_file(found_dir, "R.txt", r_lines)
    _save_vector_file(found_dir, "T.txt", t_lines)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
