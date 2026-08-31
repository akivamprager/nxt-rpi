"""Manual sanity check for depth_server.py's real model: feed it an actual
photo and look at the resulting depth map, rather than trusting a synthetic
all-zeros frame (see the pipeline-vs-accuracy caveat in midas_backend.py's
docstring — a solid-black test frame proves the wire protocol works, not
that the depth values mean anything).

    python3 pi/tools/depth_smoke_test.py path/to/photo.jpg

Requires depth_server.py already running (see its own docstring). Saves
<photo>_depth.png next to the input: a normalized grayscale visualization
where brighter = closer to the camera.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scout.depth_estimator import RemoteDepthEstimator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", help="path to a real JPEG/PNG photo")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    frame = cv2.imread(args.photo)
    if frame is None:
        print(f"could not read {args.photo}", file=sys.stderr)
        return 1

    depth = RemoteDepthEstimator(args.host, args.port).estimate(frame)
    print(f"depth shape: {depth.shape}")
    print(f"min/max/mean mm: {depth.min():.1f} / {depth.max():.1f} / {depth.mean():.1f}")

    normalized = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6)
    # Brighter = closer: invert so smaller mm (nearer) maps to a higher pixel value.
    visual = ((1.0 - normalized) * 255).astype(np.uint8)
    out_path = os.path.splitext(args.photo)[0] + "_depth.png"
    cv2.imwrite(out_path, visual)
    print(f"wrote {out_path} — brighter = closer to the camera")
    return 0


if __name__ == "__main__":
    sys.exit(main())
