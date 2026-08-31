"""The real ML backend for depth_server.py: MiDaS-small via onnxruntime.

Not imported by depth_server.py's own tests (see test_depth_server.py) —
only by depth_server.main() when actually running as a server, so the
routing/wire-protocol layer stays fully testable without onnxruntime or a
model file installed at all.

Model: midas_v21_small_256.onnx from julienkay/sentis-MiDaS
(https://huggingface.co/julienkay/sentis-MiDaS), a MiDaS v2.1 small
(EfficientNet-Lite3 encoder) export with input normalization baked into the
graph — feed it RGB pixels in [0, 1], NCHW; no external ImageNet mean/std
subtraction needed. Input/output shapes are read off the loaded model
itself rather than hardcoded (see load_estimate_fn), so this fails loudly
on a shape mismatch instead of silently producing garbage depth — this
matters here specifically because onnxruntime couldn't be installed inside
this dev sandbox (blocked by a pip SSL error through its proxy), so the
actual downloaded file has not been run end-to-end from within this
environment; run this on real hardware/your own machine to verify.

Setup:
    pip install onnxruntime
    mkdir -p pi/tools/models
    curl -L -o pi/tools/models/midas_v21_small_256.onnx \\
        https://huggingface.co/julienkay/sentis-MiDaS/resolve/main/onnx/midas_v21_small_256.onnx
    python3 pi/tools/depth_server.py
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(_HERE, "models", "midas_v21_small_256.onnx")

#: MiDaS predicts *relative* inverse depth, not metric millimetres, and its
#: raw output range varies scene to scene — confirmed empirically, not just
#: argued: a solid-black smoke-test frame and a real room photo produced
#: wildly different raw ranges, so a single fixed scale/shift constant
#: (this module's first version) mapped a real photo to an obviously wrong
#: 0.5-9.1mm. Per-frame min/max normalization into an assumed [near_mm,
#: far_mm] range sidesteps needing a universal constant at all, at a real
#: cost worth stating plainly: depth values are only self-consistent
#: WITHIN one frame, not comparable across frames/sweeps — a wall at the
#: edge of one frame's view and a closer wall filling another frame's whole
#: view could both get mapped near far_mm. Fixing that needs either a
#: metric-depth model variant, IMU/motion-based scale recovery, or
#: ArUco-anchored scale correction — real, deferred work, not a constant to
#: guess here.
DEFAULT_NEAR_MM = 150.0
DEFAULT_FAR_MM = 3000.0


def load_estimate_fn(
    model_path: str = DEFAULT_MODEL_PATH,
    near_mm: float = DEFAULT_NEAR_MM,
    far_mm: float = DEFAULT_FAR_MM,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the estimate_fn depth_server.make_handler expects. Loads the
    model once, here, at server startup — not per-request."""
    import onnxruntime as ort

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"MiDaS model not found at {model_path} — see this module's "
            f"docstring for the download command."
        )

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    input_name = input_meta.name
    # Static NCHW shape, e.g. (1, 3, 256, 256) — read from the model rather
    # than hardcoded, so a differently-sized export (384/512) still works.
    _, channels, in_h, in_w = input_meta.shape
    if channels != 3:
        raise ValueError(f"expected a 3-channel RGB input, model declares {channels}")

    def estimate(frame_bgr: np.ndarray) -> np.ndarray:
        import cv2

        orig_h, orig_w = frame_bgr.shape[:2]
        resized = cv2.resize(frame_bgr, (in_w, in_h))
        rgb01 = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = np.transpose(rgb01, (2, 0, 1))[np.newaxis, ...]  # HWC -> NCHW

        (raw,) = session.run([output_name], {input_name: input_tensor})
        raw = np.squeeze(raw)
        if raw.ndim != 2:
            raise ValueError(f"expected a 2D depth map from the model, got shape {raw.shape}")

        # Larger raw value = nearer (relative inverse depth) — see the
        # module-level note. Map [raw.min(), raw.max()] onto [far_mm,
        # near_mm] (note the order: max raw -> near_mm).
        raw64 = raw.astype(np.float64)
        raw_min, raw_max = raw64.min(), raw64.max()
        if raw_max - raw_min < 1e-9:
            # A content-free frame (e.g. solid black) gives no relative
            # signal at all — fall back to the midpoint rather than
            # dividing by ~zero and producing huge/garbage values.
            metric_mm = np.full(raw64.shape, (near_mm + far_mm) / 2.0)
        else:
            normalized = (raw64 - raw_min) / (raw_max - raw_min)  # 0=far, 1=near
            metric_mm = far_mm - normalized * (far_mm - near_mm)

        return cv2.resize(metric_mm.astype(np.float32), (orig_w, orig_h))

    return estimate
