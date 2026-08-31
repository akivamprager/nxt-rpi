"""Pi Camera Module capture — the piece vision.py already assumes exists
("the camera capture loop is responsible for the color/format conversion —
see the note in capture.py about what picamera2 hands back") but that
hadn't been built until now.

picamera2 is NOT installed via pip on the Pi — it ships with Raspberry Pi OS
Bookworm (see requirements.txt): `sudo apt install -y python3-picamera2`.
Imported lazily inside PiCamera.__init__, never at module import time, so
importing this module — or anything that imports it, like depth_estimator's
real-hardware wiring — never requires picamera2 to be installed anywhere
except the real Pi. That also means this class can't be exercised from any
dev machine, only real hardware, the same limitation transport.py's
BluetoothTransport already carries.

picamera2's `.capture_array()` hands back RGB888 by default (see its own
docs) — this module does the one color conversion each caller actually
needs (vision.ArucoDetector wants grayscale; depth_estimator.DepthEstimator
and the depth-scanner wiring want BGR, matching cv2's usual convention)
rather than pushing that choice onto every caller individually.
"""

from __future__ import annotations

import numpy as np


class PiCamera:
    """Construct once, call capture_bgr()/capture_gray() as needed for the
    lifetime of the process — not a context manager, since the mission loop
    holds one camera open for the whole run (mission.py's depth_scanner and
    localizer hooks are both called repeatedly against the same open
    camera), not opened fresh per frame."""

    def __init__(self, resolution: tuple[int, int] = (1280, 720)) -> None:
        from picamera2 import Picamera2  # only importable on the Pi itself

        self._camera = Picamera2()
        config = self._camera.create_still_configuration(main={"size": resolution})
        self._camera.configure(config)
        self._camera.start()

    def capture_rgb(self) -> np.ndarray:
        return self._camera.capture_array()

    def capture_bgr(self) -> np.ndarray:
        import cv2

        return cv2.cvtColor(self.capture_rgb(), cv2.COLOR_RGB2BGR)

    def capture_gray(self) -> np.ndarray:
        import cv2

        return cv2.cvtColor(self.capture_rgb(), cv2.COLOR_RGB2GRAY)

    def close(self) -> None:
        self._camera.close()
