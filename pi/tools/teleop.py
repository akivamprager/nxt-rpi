"""Keyboard teleoperation and live telemetry — the Phase 2 gate.

Drive the robot by hand and watch its telemetry. If this runs cleanly for ten
minutes with no dropouts and a flat ``checksum_errors`` count, the transport,
protocol, and firmware layers are all working and you can move on to vision.

Against the real robot (on the Pi)::

    python3 pi/tools/teleop.py

Against the simulator (anywhere, no hardware)::

    python3 pi/tools/sim_firmware.py --port 5555      # in one terminal
    python3 pi/tools/teleop.py --sim                  # in another

Controls:
    W/S     forward / back          A/D   turn left / right
    Q/E     turret left / right     Z     centre turret
    Space   stop                    X     toggle the ultrasonic safety stop
    +/-     speed up / down         R     reset odometry to the origin
    Ctrl-C  quit
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import termios
import time
import tty

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from scout import protocol as p  # noqa: E402
from scout.robot import Robot, RobotError  # noqa: E402
from scout.transport import BluetoothTransport, SocketTransport  # noqa: E402

SPEEDS = [50, 100, 150, 200, 250, 300]
TURN_RATE = 60.0
TURRET_STEP = 15.0


class RawKeyboard:
    """Unbuffered, non-blocking stdin, restored on exit."""

    def __enter__(self):
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc_info):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def key(self) -> str | None:
        import select

        ready, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if ready else None


def render(robot: Robot, speed: int, safety: bool, message: str) -> None:
    telemetry = robot.telemetry
    stats = robot.link_stats()

    sys.stdout.write("\033[H\033[J")  # home + clear
    print("Scout teleop      Ctrl-C to quit")
    print("=" * 58)

    if telemetry is None:
        print("  waiting for telemetry...")
    else:
        print(f"  pose      x={telemetry.x_mm:8.0f} mm  y={telemetry.y_mm:8.0f} mm"
              f"  hdg={telemetry.heading_deg:7.1f}deg")
        print(f"  turret    {telemetry.turret_deg:7.1f} deg")

        if telemetry.has_echo:
            near = telemetry.range_cm <= 20
            print(f"  sonar     {telemetry.range_cm:3d} cm"
                  f"{'   <-- OBSTACLE' if near else ''}")
        else:
            # Not the same as 'clear' — the beam simply did not come back.
            print("  sonar     no echo (unknown, not clear)")

        print(f"  colour    id={telemetry.color_id}  light={telemetry.light}")
        print(f"  bumpers   {'PRESSED' if telemetry.bumper_pressed else 'clear'}")
        print(f"  battery   {telemetry.battery_mv/1000.0:.2f} V")

        flags = []
        if telemetry.moving:
            flags.append("moving")
        if telemetry.flags & p.FLAG_TURRET_MOVING:
            flags.append("turret")
        if telemetry.safety_tripped:
            flags.append("SAFETY TRIPPED")
        print(f"  state     {', '.join(flags) if flags else 'idle'}")

    print("-" * 58)
    print(f"  speed {speed} mm/s     safety {'on' if safety else 'OFF'}")

    # A climbing checksum count is the signature of the WiFi/Bluetooth
    # coexistence problem, so it is surfaced rather than hidden.
    warn = "  <-- link degrading" if stats["checksum_errors"] else ""
    print(f"  frames {stats['frames_received']}   checksum errors "
          f"{stats['checksum_errors']}{warn}")
    print(f"  telemetry age {stats['telemetry_age_s']:.2f} s")
    print("-" * 58)
    print("  W/S drive   A/D turn   Q/E turret   Z centre")
    print("  Space stop  X safety   +/- speed    R reset odometry")
    if message:
        print(f"\n  {message}")


def _connect_with_retry(
    host: str, port: int, timeout: float = 5.0, poll_interval: float = 0.1
) -> socket.socket:
    """Connect to the simulator, tolerating the startup race.

    ``sim_firmware.py --port N &`` followed immediately by ``teleop.py --sim``
    on the next shell line is a common way to run these two together, and the
    simulator's listen socket is not always bound yet when the client tries to
    connect. A bare ``ConnectionRefusedError`` there is a false alarm, not a
    real failure, so this retries instead of giving up on the first attempt.
    """
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            return socket.create_connection((host, port), timeout=1.0)
        except ConnectionRefusedError as exc:
            last_error = exc
            time.sleep(poll_interval)
    raise ConnectionRefusedError(
        f"could not reach the simulator at {host}:{port} after {timeout:.0f}s "
        f"of retrying. Is 'python3 pi/tools/sim_firmware.py --port {port}' "
        f"actually running?"
    ) from last_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", action="store_true",
                        help="connect to the simulator instead of real hardware")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="/dev/rfcomm0")
    args = parser.parse_args()

    if args.sim:
        print(f"Connecting to the simulator at {args.host}:{args.port} ...")
        sock = _connect_with_retry(args.host, args.port)
        transport = SocketTransport(sock)
    else:
        transport = BluetoothTransport(args.device)
        print(f"Connecting to the NXT via {args.device} ...")

    robot = Robot(transport, telemetry_period_ms=100)
    try:
        robot.connect(timeout=5.0)
    except (RobotError, IOError) as exc:
        print(f"\nCould not connect: {exc}")
        return 1

    speed_index = 2
    safety = True
    message = ""
    message_until = 0.0

    try:
        with RawKeyboard() as keyboard:
            while True:
                key = keyboard.key()
                if key:
                    key = key.lower()
                    try:
                        if key == "w":
                            if not robot.drive(SPEEDS[speed_index], 0):
                                message, message_until = (
                                    "refused: obstacle ahead", time.time() + 2)
                        elif key == "s":
                            robot.drive(-SPEEDS[speed_index], 0)
                        elif key == "a":
                            robot.drive(0, TURN_RATE)
                        elif key == "d":
                            robot.drive(0, -TURN_RATE)
                        elif key == " ":
                            robot.stop()
                        elif key in ("q", "e"):
                            telemetry = robot.telemetry
                            current = telemetry.turret_deg if telemetry else 0.0
                            step = TURRET_STEP if key == "q" else -TURRET_STEP
                            robot.turret_to(current + step, wait=False)
                        elif key == "z":
                            robot.turret_to(0.0, wait=False)
                        elif key == "x":
                            safety = not safety
                            robot.set_safety(safety)
                            message, message_until = (
                                f"ultrasonic safety {'on' if safety else 'OFF'} "
                                f"(bumpers always stop the robot)",
                                time.time() + 3)
                        elif key == "r":
                            robot.set_pose(0, 0, 0)
                            message, message_until = (
                                "odometry reset to origin", time.time() + 2)
                        elif key in ("+", "="):
                            speed_index = min(speed_index + 1, len(SPEEDS) - 1)
                        elif key in ("-", "_"):
                            speed_index = max(speed_index - 1, 0)
                    except RobotError as exc:
                        message, message_until = f"command failed: {exc}", time.time() + 3

                if time.time() > message_until:
                    message = ""

                if not robot.connected:
                    print("\n\nLink lost.")
                    return 1

                render(robot, SPEEDS[speed_index], safety, message)
                time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            robot.stop()
        except (RobotError, IOError):
            pass
        robot.close()
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
