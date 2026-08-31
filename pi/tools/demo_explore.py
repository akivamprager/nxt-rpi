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
                  local interactive use doesn't need this. Independent of
                  the dashboard's own Reset button (POST /reset_mission,
                  always available): that lets a viewer restart on demand
                  at any point, not just once a mission happens to finish
                  — the coverage % dropping suddenly on a public demo with
                  SCOUT_LOOP set is this feature doing exactly what it's
                  for (a fresh lap starting), not a bug, though it can look
                  like one without the "Starting a fresh lap" log line
                  visible.
    MESH_RECONSTRUCT_PYTHON
                  path to a Python interpreter with Open3D installed (e.g.
                  .venv-mesh/bin/python3 — see mesh_reconstruct.py's
                  docstring for why that's usually a separate Python from
                  this one). Powers scene.html's "download the accurate
                  mesh" button. Unset by default: that button then 503s
                  with a clear message rather than failing silently — a
                  public deployment (e.g. Render) won't have this
                  configured unless specifically set up for it.
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
from scout.mission import ExplorationMission  # noqa: E402
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

    # Box-shaped stand-ins for scene.html's decorative furniture (sofa,
    # coffee table, bookshelf, floor lamp — the rug is skipped: it's flush
    # with the floor, not a real 3D obstacle a depth ray would distinguish
    # from the floor itself). Positions/colours hand-converted from
    # scene.html's furnish() layout for THIS room's exact wall bounds (see
    # that function's own minX/maxX/minZ/maxZ + marginFromWall math) so the
    # depth-scanned point cloud/mesh actually matches what the 3D scene
    # view renders, not just an empty shell — see FurnitureBox's docstring
    # for why this only affects depth_scan, not 2D collision/exploration.
    room.add_furniture(  # sofa: scene.html's fabric 0x5c6f8a
        -600.0, 850.0, width_x_mm=900.0, depth_y_mm=450.0, height_mm=440.0,
        color=(92, 111, 138),
    )
    room.add_furniture(  # coffee table: scene.html's wood 0x8a6540
        -600.0, 300.0, width_x_mm=500.0, depth_y_mm=280.0, height_mm=270.0,
        color=(138, 101, 64),
    )
    room.add_furniture(  # bookshelf: scene.html's wood 0x5a4632, rotated 90deg
        1150.0, -600.0, width_x_mm=240.0, depth_y_mm=500.0, height_mm=1100.0,
        color=(90, 70, 50),
    )
    # Floor lamp: scene.html's metal 0x2b2e33. A coarse stand-in — its real
    # shape (a thin ~10mm pole with a wide shade near the top) has no good
    # single-box approximation, so this is sized more like the shade+base
    # region than the whole fixture.
    room.add_furniture(
        1045.0, 745.0, width_x_mm=200.0, depth_y_mm=200.0, height_mm=1500.0,
        color=(43, 46, 51),
    )
    return room


#: A slight downward tilt for the simulated depth camera, matching the
#: visual forward-leaning bracket in scene.html's robot model. Originally
#: -20 degrees with a narrower 45-degree vertical FOV — heavily floor-
#: biased, which produced a point cloud only 0-260mm tall (verified against
#: a real mesh_reconstruct.py run: Poisson reconstruction of that thin,
#: non-enclosing "shell" collapsed into a twisted-ribbon artifact instead of
#: a room, since it assumes a proper enclosed 3D surface). Widened to -5/90
#: below so a scan's vertical cone actually spans floor to above the room's
#: 900mm wall height (see SimulatedRoom.wall_height_mm) across the range of
#: distances a robot sees walls from, not just the ground right in front of
#: it — the difference between "floor/obstacle mapping" and "reconstructing
#: the room's shape," which is what this demo is actually for.
DEPTH_CAMERA_PITCH_DEG = -5.0
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
            v_fov_deg=90.0,
            h_samples=14,
            v_samples=16,
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

    # Set from the dashboard's HTTP thread (POST /reset_mission -> reset_fn
    # below) to ask the main loop, which owns the actual mission lifecycle,
    # to restart. threading.Event is the right primitive here specifically
    # because it's safe to .set() from any thread and the main loop can
    # .wait()/.is_set() on it without polling in a busy loop.
    restart_event = threading.Event()

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
        mesh_reconstruct_python=os.environ.get("MESH_RECONSTRUCT_PYTHON"),
        reset_fn=restart_event.set,
    )
    print("Scout is exploring a simulated room.")
    print(f"Open http://{host}:{port} for the 2D map, or")
    print(f"     http://{host}:{port}/scene.html for the 3D view.")
    if loop:
        print("SCOUT_LOOP is set: a new lap starts automatically once one finishes.")
    print("A Reset button in the dashboard restarts exploration on demand, any time.")
    print("Ctrl-C to stop.\n")

    try:
        while True:
            mission = current["mission"]
            mission_thread = threading.Thread(target=mission.run, daemon=True)
            mission_thread.start()

            # Wait for whichever comes first: the mission finishing on its
            # own, or a manual reset request. join(timeout=...) rather than
            # a plain join() specifically so this loop keeps checking
            # restart_event instead of blocking past it.
            while mission_thread.is_alive() and not restart_event.is_set():
                mission_thread.join(timeout=0.5)

            manual_reset = restart_event.is_set()
            if manual_reset:
                mission.stop()  # ask a still-running mission to wind down now
                mission_thread.join(timeout=5.0)
            restart_event.clear()

            print(
                f"Mission ended in state {mission.state} — grid coverage "
                f"{mission.grid.coverage():.0%}."
                + (" (manual reset)" if manual_reset else "")
            )

            if not manual_reset and not loop:
                print("Dashboard stays up; Ctrl-C to quit, or use the dashboard's Reset button.")
                restart_event.wait()  # blocks here until a reset request arrives
                restart_event.clear()
            elif not manual_reset:
                # Auto-loop pacing only — a manual reset means the viewer
                # explicitly asked to start over right now, so restarting
                # immediately is the more responsive, expected behavior;
                # the pause exists purely to let an auto-finished map stay
                # on screen a moment before SCOUT_LOOP sweeps it away.
                time.sleep(LOOP_PAUSE_S)

            robot.set_pose(0.0, 0.0, 0.0)
            new_mission = ExplorationMission(
                robot, new_grid(), min_frontier_cluster=2, depth_scanner=depth_scanner
            )
            current["mission"] = new_mission
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
