#!/usr/bin/env python3
"""Extract frames from a video into JPG files.

Usage example:
    python video_to_jpg.py input.mov output_dir/

The script imports OpenCV only after parsing args to allow `--help` to work even if OpenCV
is not installed.
"""

from pathlib import Path
import argparse
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Extract frames from a video into JPG files")
    p.add_argument("input",
                   help="Input video file (e.g. input.mov)")
    p.add_argument("output",
                   help="Output directory where frames will be written")
    p.add_argument("--prefix", "-p", default="frame_",
                   help="Filename prefix for saved frames (default: frame_)")
    p.add_argument("--start", "-s", type=int, default=0,
                   help="Start from this frame index (default: 0)")
    p.add_argument("--step", type=int, default=1,
                   help="Save every Nth frame (default: 1)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after saving this many frames (0 = no limit)")
    p.add_argument("--quality", type=int, default=95,
                   help="JPEG quality 0-100 (default: 95)")
    p.add_argument("--ext", choices=("jpg", "jpeg"), default="jpg",
                   help="Output image extension (default: jpg)")
    return p.parse_args()


def main():
    args = parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)

    if not in_path.exists():
        print(f"Error: input file does not exist: {in_path}")
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)

    # import cv2 only now so `--help` works without OpenCV installed
    try:
        import cv2
    except Exception as e:
        print("Error: OpenCV (cv2) is required but not installed.")
        print("Install with: pip install opencv-python")
        print(f"Import error: {e}")
        sys.exit(3)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"Error: could not open video: {in_path}")
        sys.exit(4)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    print(f"Opened video: {in_path} — frames={total}, size={width}x{height}")

    saved = 0
    idx = 0
    next_to_save = args.start

    # loop through frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx < args.start:
            idx += 1
            continue

        if (idx - args.start) % args.step != 0:
            idx += 1
            continue

        out_name = f"{args.prefix}{idx:06d}.{args.ext}"
        out_path = out_dir / out_name

        # write JPEG with requested quality
        try:
            ok = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)])
            if not ok:
                print(f"Warning: failed to write {out_path}")
            else:
                saved += 1
        except Exception as e:
            print(f"Warning: exception while writing {out_path}: {e}")

        if saved % 100 == 0 and saved > 0:
            print(f"Saved {saved} frames (last: {out_name})")

        if args.max_frames and saved >= args.max_frames:
            break

        idx += 1

    cap.release()
    print(f"Done — saved {saved} frames to: {out_dir}")


if __name__ == "__main__":
    main()
