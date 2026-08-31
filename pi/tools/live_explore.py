"""Run Scout's exploration mission on real hardware, with real depth-camera
3D mapping — the non-simulated counterpart to demo_explore.py.

Needs actual hardware: an NXT running the leJOS firmware, reachable over
Bluetooth RFCOMM, and a Pi Camera Module for depth-camera mapping (see
scout/capture.py). Nothing in this file can be exercised without that
hardware, the same limitation transport.BluetoothTransport and
capture.PiCamera already carry individually — this is where they, and
depth_estimator.py's real-hardware depth_scanner wiring, all actually get
plugged into one running mission.

ArUco marker localization (localize.py, vision.py's make_localizer) is
wired in too, alongside depth mapping — drift correction from surveyed
markers in config.yaml, whenever any are placed. With none placed
(config.yaml's `markers: []`), the mission runs on odometry alone, same as
demo_explore.py; markers only start correcting drift once you've printed,
placed, and surveyed at least one (see docs/BUILD.md).

    python3 pi/tools/live_explore.py

Environment variables:
    BT_DEVICE            RFCOMM device node for the NXT (default
                          /dev/rfcomm0, matching teleop.py's own default).
    CONFIG_PATH           path to config.yaml (default: pi/config.yaml).
    DEPTH_BACKEND         "remote" (default) or "tflite" — which
                          DepthEstimator backend powers the depth scanner.
    DEPTH_SERVER_HOST     host running depth_server.py (default 127.0.0.1),
                          only used when DEPTH_BACKEND=remote.
    DEPTH_SERVER_PORT     port depth_server.py listens on (default 8090),
                          only used when DEPTH_BACKEND=remote.
    DEPTH_MODEL_PATH      path to a .tflite model, only used when
                          DEPTH_BACKEND=tflite (see
                          scout/tflite_depth_estimator.py's setup notes).
    PORT / SCOUT_PORT     dashboard port (default 8080).
    SCOUT_HOST            dashboard bind interface (default 127.0.0.1).
"""

from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scout import config as cfg  # noqa: E402
from scout.capture import PiCamera  # noqa: E402
from scout.depth_estimator import DepthEstimator, make_depth_scanner  # noqa: E402
from scout.localize import CameraGeometry  # noqa: E402
from scout.mapping import OccupancyGrid  # noqa: E402
from scout.mission import ExplorationMission  # noqa: E402
from scout.robot import Robot  # noqa: E402
from scout.transport import BluetoothTransport  # noqa: E402
from scout.vision import ArucoDetector, CameraIntrinsics, make_localizer  # noqa: E402
from web.server import start as start_dashboard  # noqa: E402

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml"
)


def _build_depth_estimator() -> DepthEstimator:
    backend = os.environ.get("DEPTH_BACKEND", "remote")
    if backend == "remote":
        from scout.depth_estimator import RemoteDepthEstimator

        host = os.environ.get("DEPTH_SERVER_HOST", "127.0.0.1")
        port = int(os.environ.get("DEPTH_SERVER_PORT", "8090"))
        return RemoteDepthEstimator(host, port)
    if backend == "tflite":
        from scout.tflite_depth_estimator import TFLiteDepthEstimator

        model_path = os.environ.get("DEPTH_MODEL_PATH")
        if not model_path:
            raise ValueError("DEPTH_MODEL_PATH must be set when DEPTH_BACKEND=tflite")
        return TFLiteDepthEstimator(model_path)
    raise ValueError(f"unknown DEPTH_BACKEND {backend!r} (expected 'remote' or 'tflite')")


def main() -> int:
    host = os.environ.get("SCOUT_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", os.environ.get("SCOUT_PORT", "8080")))
    config_path = os.environ.get("CONFIG_PATH", _DEFAULT_CONFIG_PATH)
    bt_device = os.environ.get("BT_DEVICE", "/dev/rfcomm0")

    config = cfg.load(config_path)

    transport = BluetoothTransport(bt_device)
    print(f"Connecting to the NXT via {bt_device} ...")
    robot = Robot(transport, telemetry_period_ms=100)
    robot.connect(timeout=5.0)
    robot.wait_for_telemetry(timeout=2.0)

    camera = PiCamera()
    intrinsics = CameraIntrinsics.load(
        os.path.join(os.path.dirname(config_path), config.camera.intrinsics_path)
    )
    geometry = CameraGeometry(
        camera_mount=config.camera.camera_mount, turret_mount=config.camera.turret_mount
    )
    estimator = _build_depth_estimator()
    depth_scanner = make_depth_scanner(
        camera.capture_bgr, estimator, intrinsics, geometry,
        camera_height_mm=config.camera.camera_height_mm,
        camera_pitch_deg=config.camera.camera_pitch_deg,
    )

    localizer = None
    if config.markers:
        detector = ArucoDetector(intrinsics, config.camera.marker_size_mm)
        localizer = make_localizer(camera.capture_gray, detector, config.markers, geometry)
    else:
        print("No surveyed markers in config.yaml — running on odometry alone.")

    grid = OccupancyGrid(
        width=config.grid.width, height=config.grid.height,
        cell_size_mm=config.grid.cell_size_mm,
        origin_x_mm=config.grid.origin_x_mm, origin_y_mm=config.grid.origin_y_mm,
    )
    mission = ExplorationMission(
        robot, grid,
        sweep_angles=config.mission.sweep_angles_deg,
        travel_step_mm=config.mission.travel_step_mm,
        min_frontier_cluster=config.mission.min_frontier_cluster,
        depth_scanner=depth_scanner,
        localizer=localizer,
    )

    dashboard = start_dashboard(
        mission.snapshot,
        pointcloud_fn=lambda: mission.point_cloud.to_dict(),
        host=host, port=port,
    )
    print("Scout is exploring — real hardware, real depth camera.")
    print(f"Open http://{host}:{port} for the 2D map, or")
    print(f"     http://{host}:{port}/scene.html for the 3D view (no ground-truth")
    print("     walls on real hardware — only the earned point cloud).")
    print("Ctrl-C to stop.\n")

    mission_thread = threading.Thread(target=mission.run, daemon=True)
    mission_thread.start()
    try:
        mission_thread.join()
        print(f"Mission ended in state {mission.state} — grid coverage {mission.grid.coverage():.0%}.")
        print("Dashboard stays up; Ctrl-C to quit.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        mission.stop()
        dashboard.shutdown()
        try:
            robot.stop()
        except Exception:  # noqa: BLE001 - best-effort on the way out
            pass
        robot.close()
        camera.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
