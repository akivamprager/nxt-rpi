"""Watch Scout autonomously explore a simulated room — no hardware at all.

This is the one script that ties everything from Phases 1-4 together: a
simulated NXT brick and a simulated room, the real Robot/mission code that
will eventually run against actual hardware unchanged, an occupancy grid,
and a live browser dashboard. Run it and open the printed URL.

    python3 pi/tools/demo_explore.py

Ctrl-C to stop. Standard library only — nothing to install.
"""

from __future__ import annotations

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


def main() -> int:
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

    # Sized to cover the room with margin on all sides, per BUILD.md's note
    # that the grid represents a bounded area — the edge of the array is a
    # modelling boundary, not something to explore toward.
    grid = OccupancyGrid(
        width=34, height=28, cell_size_mm=100.0, origin_x_mm=-1700.0, origin_y_mm=-1400.0
    )
    mission = ExplorationMission(robot, grid, min_frontier_cluster=2)

    mission_thread = threading.Thread(target=mission.run, daemon=True)
    mission_thread.start()

    def room_fn() -> dict:
        data = room.to_dict()
        data["robot"] = load_wheel_geometry()
        return data

    dashboard = start_dashboard(mission.snapshot, room_fn=room_fn, port=8080)
    print("Scout is exploring a simulated room.")
    print("Open http://127.0.0.1:8080 for the 2D map, or")
    print("     http://127.0.0.1:8080/scene.html for the 3D view.")
    print("Ctrl-C to stop.\n")

    try:
        while mission_thread.is_alive():
            time.sleep(0.5)
        print(
            f"Mission ended in state {mission.state} — grid coverage "
            f"{grid.coverage():.0%}. Dashboard stays up; Ctrl-C to quit."
        )
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        mission.stop()
        mission_thread.join(timeout=5.0)
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
