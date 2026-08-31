"""Watch Scout autonomously explore a simulated room — no hardware at all.

This is the one script that ties everything from Phases 1-4 together: a
simulated NXT brick and a simulated room, the real Robot/mission code that
will eventually run against actual hardware unchanged, an occupancy grid,
and a live browser dashboard. Run it and open the printed URL.

    python3 pi/tools/demo_explore.py

Ctrl-C to stop. Standard library only — nothing to install, confirmed by
tracing the actual import chain (see docs/DEPLOY.md): config.yaml parsing is
the one place PyYAML would matter, and that import is lazy with a safe
fallback (load_wheel_geometry below), so even a bare `python3` with nothing
pip-installed runs this correctly.

Environment variables (all optional, all default to the plain local-dev
behavior above):

    PORT          port to listen on (default 8080). Read as `PORT` first,
                  matching what most hosting platforms inject, before
                  falling back to `SCOUT_PORT` for a locally-meaningful name.
    SCOUT_HOST    interface to bind (default 127.0.0.1). Public deployment
                  needs 0.0.0.0 — see docs/DEPLOY.md's render.yaml, which
                  sets this explicitly rather than defaulting a demo tool to
                  a publicly-reachable bind.
    SCOUT_LOOP    if set, start a fresh lap (new grid, robot pose reset to
                  the origin) once a mission reaches DONE, instead of
                  stopping there. A public demo that just goes still after
                  finishing is a worse demo than one that keeps exploring;
                  local interactive use doesn't need this.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scout import config as cfg  # noqa: E402
from scout.mapping import OccupancyGrid  # noqa: E402
from scout.mission import DONE, ExplorationMission  # noqa: E402
from scout.robot import Robot  # noqa: E402
from scout.transport import SocketTransport  # noqa: E402
from sim_firmware import make_simulated_pair  # noqa: E402
from sim_world import SimulatedRoom  # noqa: E402
from web.server import start as start_dashboard  # noqa: E402

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")

#: How long the finished map stays on screen before a fresh lap starts, when
#: SCOUT_LOOP is set — long enough that a viewer who just watched it finish
#: can actually see the completed map, short enough not to look stalled.
LOOP_PAUSE_S = 8.0


def load_wheel_geometry() -> dict:
    """Real measurements from config.yaml if it parses, otherwise a
    reasonable fallback for the demo. Ties the 3D scene's wheel size and
    spacing to the same source that will drive real motion once hardware
    exists, per docs/BUILD.md's "before Phase 1" measurements."""
    try:
        robot = cfg.load(_CONFIG_PATH).robot
        return {
            "wheel_diameter_mm": robot.wheel_diameter_mm,
            "track_width_mm": robot.track_width_mm,
        }
    except Exception:  # noqa: BLE001 - config.yaml may not parse yet; demo still runs
        return {"wheel_diameter_mm": 56.0, "track_width_mm": 115.0}


def build_room() -> SimulatedRoom:
    """A modest 3m x 2.4m room with a partial interior divider, so
    exploration has to route around something rather than just mapping an
    empty box. Centered on the origin — the robot's odometry always starts
    at (0, 0), so an off-center room would have it starting inside a wall.
    """
    room = SimulatedRoom(
        walls=[
            ((-1500.0, -1200.0), (1500.0, -1200.0)),
            ((1500.0, -1200.0), (1500.0, 1200.0)),
            ((1500.0, 1200.0), (-1500.0, 1200.0)),
            ((-1500.0, 1200.0), (-1500.0, -1200.0)),
        ]
    )
    room.add_wall((-200.0, -1200.0), (-200.0, 300.0))
    return room


#: A fixed downward tilt for the simulated depth camera, matching the visual
#: forward-leaning bracket in scene.html's robot model — a level camera
#: would mostly see the far wall instead of the floor and nearby obstacles a
#: real mapping camera needs to see.
DEPTH_CAMERA_PITCH_DEG = -20.0
#: Mount point relative to the chassis origin: forward of the turret's own
#: axis (the camera sits at the front of the turret housing, not its
#: center) and up at roughly turret height.
DEPTH_CAMERA_FORWARD_OFFSET_MM = 40.0
DEPTH_CAMERA_HEIGHT_MM = 150.0


def make_depth_scanner(room: SimulatedRoom):
    """A closure mirroring the mission's other simulated sensors: given
    telemetry, return the points a real depth camera would have reported
    from that pose. `turret_deg` is included in yaw the same way it already
    factors into the vision/localization code — the camera rides the
    turret, so it looks wherever the turret is pointed, not just wherever
    the chassis is facing."""

    def scan(telemetry) -> list[tuple[float, float, float]]:
        yaw_deg = telemetry.heading_deg + telemetry.turret_deg
        yaw_rad = math.radians(yaw_deg)
        cam_x = telemetry.x_mm + DEPTH_CAMERA_FORWARD_OFFSET_MM * math.cos(yaw_rad)
        cam_y = telemetry.y_mm + DEPTH_CAMERA_FORWARD_OFFSET_MM * math.sin(yaw_rad)
        return room.depth_scan(
            cam_x,
            cam_y,
            DEPTH_CAMERA_HEIGHT_MM,
            yaw_deg=yaw_deg,
            pitch_deg=DEPTH_CAMERA_PITCH_DEG,
            h_fov_deg=60.0,
            v_fov_deg=45.0,
            h_samples=14,
            v_samples=10,
            max_range_mm=2500.0,
        )

    return scan


def new_grid() -> OccupancyGrid:
    # Sized to cover the room with margin on all sides, per BUILD.md's note
    # that the grid represents a bounded area — the edge of the array is a
    # modelling boundary, not something to explore toward.
    return OccupancyGrid(
        width=34, height=28, cell_size_mm=100.0, origin_x_mm=-1700.0, origin_y_mm=-1400.0
    )


def main() -> int:
    host = os.environ.get("SCOUT_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", os.environ.get("SCOUT_PORT", "8080")))
    loop = bool(os.environ.get("SCOUT_LOOP"))

    room = build_room()
    pi_sock, sim = make_simulated_pair(telemetry_period_ms=50, world=room)

    # A few times faster than real life so a full lap of the room takes
    # roughly a minute to watch instead of several. Purely a demo-pacing
    # choice — every rate is scaled together, so behavior is unchanged.
    sim.travel_speed *= 4
    sim.rotate_speed *= 4
    sim.turret_speed *= 4

    robot = Robot(SocketTransport(pi_sock), telemetry_period_ms=50)
    robot.connect(timeout=5.0)
    robot.wait_for_telemetry(timeout=2.0)

    # A mutable box, not a bare variable: the dashboard's snapshot_fn closes
    # over this so it always reads whichever mission is current, even after
    # SCOUT_LOOP swaps in a fresh one — a plain closure over `mission` would
    # keep calling the FIRST lap's (by-then finished, DONE-forever) mission.
    depth_scanner = make_depth_scanner(room)
    current = {
        "mission": ExplorationMission(
            robot, new_grid(), min_frontier_cluster=2, depth_scanner=depth_scanner
        )
    }

    def room_fn() -> dict:
        data = room.to_dict()
        data["robot"] = load_wheel_geometry()
        return data

    dashboard = start_dashboard(
        lambda: current["mission"].snapshot(),
        room_fn=room_fn,
        pointcloud_fn=lambda: current["mission"].point_cloud.to_dict(),
        host=host,
        port=port,
    )
    print("Scout is exploring a simulated room.")
    print(f"Open http://{host}:{port} for the 2D map, or")
    print(f"     http://{host}:{port}/scene.html for the 3D view.")
    if loop:
        print("SCOUT_LOOP is set: a new lap starts automatically once one finishes.")
    print("Ctrl-C to stop.\n")

    try:
        while True:
            mission = current["mission"]
            mission_thread = threading.Thread(target=mission.run, daemon=True)
            mission_thread.start()
            mission_thread.join()

            print(
                f"Mission ended in state {mission.state} — grid coverage "
                f"{mission.grid.coverage():.0%}."
            )
            if not loop:
                print("Dashboard stays up; Ctrl-C to quit.")
                while True:
                    time.sleep(1.0)

            time.sleep(LOOP_PAUSE_S)
            robot.set_pose(0.0, 0.0, 0.0)
            new_mission = ExplorationMission(
                robot, new_grid(), min_frontier_cluster=2, depth_scanner=depth_scanner
            )
            current["mission"] = new_mission
            if mission.state == DONE:
                print("Starting a fresh lap.\n")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        current["mission"].stop()
        dashboard.shutdown()
        try:
            robot.stop()
        except Exception:  # noqa: BLE001 - best-effort on the way out
            pass
        robot.close()
        sim.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
