"""On-device depth estimation for the Pi 3B itself — the other half of the
"build both efficiently" plan, alongside RemoteDepthEstimator in
depth_estimator.py.

Unverified by design, and honestly so: this needs `tflite-runtime` (or
`tensorflow.lite`) and a converted MiDaS-small `.tflite` model file, both of
which are ARM/Pi-specific — there is no way to exercise this module's actual
inference path from this dev sandbox (a different OS and CPU architecture
from the Pi 3B), the same limitation the NXT firmware and camera calibration
code already carry until real hardware exists (see docs/BUILD.md). What IS
verified here: the interface shape matches depth_estimator.DepthEstimator
exactly, so mission.py, demo_explore.py, and everything downstream can treat
this and RemoteDepthEstimator as interchangeable without caring which one is
actually plugged in.

Setup, once there's a real Pi:
    pip install tflite-runtime
    Convert or download a MiDaS-small .tflite model (see
    https://github.com/isl-org/MiDaS for the reference model and conversion
    scripts) to some path, then:
        TFLiteDepthEstimator("midas_small.tflite")

Calibration caveat (the load-bearing one): MiDaS predicts *relative* inverse
depth — good for "which pixels are closer," not metric millimetres out of
the box. `depth_scale_mm`/`depth_shift` below convert the model's raw output
into approximate metric depth via `metric = depth_scale_mm / (raw +
depth_shift)`, but the right values for those two constants depend on the
specific converted model and this camera's mount — they need calibrating
against 1-2 known real-world distances once the camera exists (same
"measure once you have hardware" pattern as everything else in config.yaml),
not guessed here. The placeholders below are not measurements.
"""

from __future__ import annotations

import numpy as np

#: Most published MiDaS-small conversions expect a 256x256 RGB input,
#: ImageNet-normalized. If the specific model file in use expects something
#: different, override at construction time rather than editing this.
DEFAULT_INPUT_SIZE = (256, 256)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TFLiteDepthEstimator:
    """A DepthEstimator (see depth_estimator.py's Protocol) backed by a
    local .tflite model — no network call, unlike RemoteDepthEstimator, at
    the cost of running on the Pi 3B's own weak, GPU-less CPU. Expect
    roughly 1-3 seconds per frame at 256x256, not real-time — acceptable
    only because mission.py's depth_scanner hook fires once per sweep, not
    per video frame (see mission.py's turn-based SWEEPING/PLANNING/DRIVING
    cycle).
    """

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        depth_scale_mm: float = 1000.0,
        depth_shift: float = 0.1,
    ) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.depth_scale_mm = depth_scale_mm
        self.depth_shift = depth_shift
        self._interpreter = None  # loaded lazily on first estimate() call

    def _ensure_loaded(self):
        if self._interpreter is not None:
            return
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter  # type: ignore[no-redef]

        interpreter = Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()
        self._interpreter = interpreter
        self._input_index = interpreter.get_input_details()[0]["index"]
        self._output_index = interpreter.get_output_details()[0]["index"]

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2  # lazy: only needed for resize/color conversion

        self._ensure_loaded()
        orig_h, orig_w = frame_bgr.shape[:2]

        resized = cv2.resize(frame_bgr, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        input_tensor = np.expand_dims(normalized, axis=0).astype(np.float32)

        self._interpreter.set_tensor(self._input_index, input_tensor)
        self._interpreter.invoke()
        raw = self._interpreter.get_tensor(self._output_index)[0]
        if raw.ndim == 3:
            raw = raw[..., 0]  # drop a trailing channel dim, if the model has one

        metric_mm = self.depth_scale_mm / (raw.astype(np.float64) + self.depth_shift)
        return cv2.resize(metric_mm.astype(np.float32), (orig_w, orig_h))
