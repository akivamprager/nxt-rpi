"""A Python stand-in for the NXT's ScoutServer firmware.

This mirrors the behaviour of ``nxt-firmware/src/scout/ScoutServer.java``
closely enough to exercise the entire Pi stack — transport, protocol, robot
API, and eventually the mission state machine — with no brick attached.

It is a development aid, not a physics simulator. Motion is modelled as simple
kinematics at constant speed. What it *does* reproduce faithfully is the wire
protocol, the ACK/event semantics, and the local safety-stop behaviour, since
those are what the Pi code has to get right.

Run standalone to drive a fake robot over TCP::

    python3 pi/tools/sim_firmware.py --port 5555
"""

from __future__ import annotations

import math
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scout import protocol as p  # noqa: E402


class SimulatedNXT:
    """Speaks the Scout protocol over a socket, pretending to be the brick."""

    def __init__(
        self,
        sock: socket.socket,
        telemetry_period_ms: int = 100,
        world: "SimulatedRoom | None" = None,
    ) -> None:
        """`world`, if given, makes the ultrasonic and bumpers real.

        Without a world (the default), `range_cm`/`bumpers` are whatever a
        test sets them to directly — this is what test_robot.py relies on to
        stage specific scenarios, so it must keep working unchanged.

        With a world, every tick overwrites `range_cm` from a ray-cast in
        the direction the turret is pointing, and `bumpers` from actual
        proximity to a wall in the direction the chassis is moving — so
        driving into a real wall produces a real bumper event, which is what
        the exploration demo needs.
        """
        self._sock = sock
        self._decoder = p.FrameDecoder()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.world = world
        #: Distance (mm) at which the chassis is considered to be touching a
        #: wall dead ahead — physical contact, not the ultrasonic's longer-
        #: range safety threshold.
        self.bump_contact_mm = 60.0
        #: Approximate half-width of the chassis, for keeping it from
        #: clipping through a wall within a single integration step.
        self.robot_radius_mm = 90.0

        # Pose in the odometry frame: millimetres and degrees.
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.turret = 0.0

        self.turret_target = 0.0
        self.turret_max = 120  # matches TURRET_MAX_ANGLE_DEG in the firmware

        self.travel_speed = 150.0  # mm/s
        self.rotate_speed = 60.0   # deg/s
        self.turret_speed = 90.0   # deg/s

        self._move_kind: str | None = None
        self._move_remaining = 0.0
        self._drive_v = 0.0
        self._drive_omega = 0.0

        # Sensors. Tests set these directly to stage a scenario.
        self.range_cm = 200
        self.color_id = 0
        self.light = 50
        self.bumpers = 0
        self.battery_mv = 8200

        self.safety_enabled = True
        self.safety_range_cm = 20
        self.telemetry_period_ms = telemetry_period_ms

        self.safety_tripped = False
        self.seq = 0

        #: Commands received, for assertions in tests.
        self.command_log: list[int] = []

    # ------------------------------------------------------------------- runtime

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="sim-nxt", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        self._sock.settimeout(0.02)
        last_tick = time.monotonic()
        next_telemetry = 0.0

        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(4096)
                if not chunk:
                    return
                for frame_type, payload in self._decoder.feed(chunk):
                    self._dispatch(frame_type, payload)
            except socket.timeout:
                pass
            except OSError:
                return

            now = time.monotonic()
            self._integrate(now - last_tick)
            last_tick = now

            if self.telemetry_period_ms > 0 and now >= next_telemetry:
                next_telemetry = now + self.telemetry_period_ms / 1000.0
                if not self._send(p.encode(p.RSP_TELEMETRY, self._telemetry().encode())):
                    return

    # ------------------------------------------------------------------ motion

    def _integrate(self, dt: float) -> None:
        if dt <= 0:
            return
        with self._lock:
            self._integrate_turret(dt)
            self._sense_world()
            self._enforce_safety()

            if self._move_kind == "travel":
                step = min(self.travel_speed * dt, abs(self._move_remaining))
                direction = 1.0 if self._move_remaining > 0 else -1.0
                self._advance(direction * step)
                self._move_remaining -= direction * step
                if abs(self._move_remaining) < 1e-6:
                    self._finish_move()

            elif self._move_kind == "rotate":
                step = min(self.rotate_speed * dt, abs(self._move_remaining))
                direction = 1.0 if self._move_remaining > 0 else -1.0
                self.heading = _normalise(self.heading + direction * step)
                self._move_remaining -= direction * step
                if abs(self._move_remaining) < 1e-6:
                    self._finish_move()

            elif self._move_kind == "drive":
                self._advance(self._drive_v * dt)
                self.heading = _normalise(self.heading + self._drive_omega * dt)

    def _advance(self, distance: float) -> None:
        if self.world is not None and distance != 0:
            # Clamp so a single integration step can't tunnel through a wall
            # the bumper hasn't had a chance to react to yet. Cast in the
            # actual direction of travel — forward and reverse look at
            # different walls, and neither is necessarily the wall nearest
            # the ultrasonic's turret-relative bearing.
            travel_bearing = self.heading if distance > 0 else _normalise(self.heading + 180.0)
            ahead = self.world.cast_mm(self.x, self.y, travel_bearing)
            if ahead is not None:
                limit = max(0.0, ahead - self.robot_radius_mm)
                if abs(distance) > limit:
                    distance = limit if distance > 0 else -limit

        radians = math.radians(self.heading)
        self.x += distance * math.cos(radians)
        self.y += distance * math.sin(radians)

    def _sense_world(self) -> None:
        """When a world is attached, make range_cm and bumpers reflect it.

        Without a world, these stay exactly whatever a test set them to —
        see the constructor docstring.
        """
        if self.world is None:
            return

        sensor_bearing = _normalise(self.heading + self.turret)
        range_cm, has_echo = self.world.sense(self.x, self.y, sensor_bearing)
        self.range_cm = range_cm if has_echo else p.US_NO_ECHO

        # Bumper contact is about the chassis, which faces `self.heading` —
        # not wherever the turret happens to be pointing.
        ahead_mm = self.world.cast_mm(self.x, self.y, self.heading)
        self.bumpers = (
            0b11
            if ahead_mm is not None and ahead_mm <= self.robot_radius_mm + self.bump_contact_mm
            else 0
        )

    def _integrate_turret(self, dt: float) -> None:
        delta = self.turret_target - self.turret
        if abs(delta) < 1e-6:
            return
        step = min(self.turret_speed * dt, abs(delta))
        self.turret += step * (1.0 if delta > 0 else -1.0)
        if abs(self.turret_target - self.turret) < 1e-6:
            self.turret = self.turret_target
            self._send(p.encode(p.RSP_EVENT, bytes([p.EV_TURRET_DONE])))

    def _finish_move(self) -> None:
        self._move_kind = None
        self._move_remaining = 0.0
        self._send(p.encode(p.RSP_EVENT, bytes([p.EV_MOVE_DONE])))

    def _moving_forward(self) -> bool:
        if self._move_kind == "travel":
            return self._move_remaining > 0
        if self._move_kind == "drive":
            return self._drive_v > 0
        return False

    def _enforce_safety(self) -> None:
        """Local safety stop — mirrors enforceSafety() in the firmware."""
        if self.bumpers:
            if self._moving_forward():
                self._halt()
                self._send(p.encode(p.RSP_EVENT, bytes([p.EV_BUMPER])))
            self.safety_tripped = True
            return

        blocked = (
            self.safety_enabled
            and self.range_cm != p.US_NO_ECHO
            and self.range_cm <= self.safety_range_cm
        )
        if blocked and self._moving_forward():
            self._halt()
            self._send(p.encode(p.RSP_EVENT, bytes([p.EV_SAFETY_STOP])))
        self.safety_tripped = blocked

    def _halt(self) -> None:
        self._move_kind = None
        self._move_remaining = 0.0
        self._drive_v = 0.0
        self._drive_omega = 0.0

    def _can_move_forward(self) -> bool:
        if self.bumpers:
            return False
        return not (
            self.safety_enabled
            and self.range_cm != p.US_NO_ECHO
            and self.range_cm <= self.safety_range_cm
        )

    # ---------------------------------------------------------------- dispatch

    def _dispatch(self, frame_type: int, payload: bytes) -> None:
        self.command_log.append(frame_type)
        import struct

        if frame_type == p.CMD_PING:
            self._send(p.encode(p.RSP_PONG))
            return

        if frame_type == p.CMD_STOP:
            with self._lock:
                self._halt()
            self._ack(frame_type, p.ACK_OK)

        elif frame_type == p.CMD_DRIVE:
            v, omega_d = struct.unpack(">hh", payload)
            if v > 0 and not self._can_move_forward():
                self._ack(frame_type, p.ACK_REFUSED)
                return
            with self._lock:
                self._move_kind = "drive"
                self._drive_v = float(v)
                self._drive_omega = omega_d / 10.0
            self._ack(frame_type, p.ACK_OK)

        elif frame_type == p.CMD_TRAVEL:
            (distance,) = struct.unpack(">i", payload)
            if distance > 0 and not self._can_move_forward():
                self._ack(frame_type, p.ACK_REFUSED)
                return
            with self._lock:
                self._move_kind = "travel"
                self._move_remaining = float(distance)
            self._ack(frame_type, p.ACK_OK)

        elif frame_type == p.CMD_TURN_TO:
            (heading_d,) = struct.unpack(">h", payload)
            with self._lock:
                self._move_kind = "rotate"
                self._move_remaining = _normalise(heading_d / 10.0 - self.heading)
            self._ack(frame_type, p.ACK_OK)

        elif frame_type == p.CMD_TURRET_TO:
            (angle_d,) = struct.unpack(">h", payload)
            requested = angle_d / 10.0
            with self._lock:
                # Clamped, exactly as the firmware does, to protect the ribbon.
                self.turret_target = max(-self.turret_max, min(self.turret_max, requested))
            self._ack(frame_type, p.ACK_OK)

        elif frame_type == p.CMD_SET_POSE:
            x, y, heading_d = struct.unpack(">iih", payload)
            with self._lock:
                self.x, self.y, self.heading = float(x), float(y), heading_d / 10.0
            self._ack(frame_type, p.ACK_OK)

        elif frame_type == p.CMD_SET_SAFETY:
            enabled, min_range = struct.unpack(">BB", payload)
            self.safety_enabled = bool(enabled)
            self.safety_range_cm = min_range
            self._ack(frame_type, p.ACK_OK)

        elif frame_type == p.CMD_SET_TELEMETRY:
            (period,) = struct.unpack(">H", payload)
            self.telemetry_period_ms = period
            self._ack(frame_type, p.ACK_OK)

        else:
            self._ack(frame_type, p.ACK_UNKNOWN_CMD)

    def _ack(self, command: int, status: int) -> None:
        self._send(p.encode(p.RSP_ACK, bytes([command & 0xFF, status])))

    def _telemetry(self) -> p.Telemetry:
        with self._lock:
            flags = 0
            if self._move_kind is not None:
                flags |= p.FLAG_MOVING
            if abs(self.turret_target - self.turret) > 1e-6:
                flags |= p.FLAG_TURRET_MOVING
            if self.safety_tripped:
                flags |= p.FLAG_SAFETY_TRIPPED
            if self.safety_enabled:
                flags |= p.FLAG_SAFETY_ENABLED

            self.seq = (self.seq + 1) & 0xFFFF
            return p.Telemetry(
                seq=self.seq,
                x_mm=self.x,
                y_mm=self.y,
                heading_deg=_normalise(self.heading),
                turret_deg=self.turret,
                range_cm=self.range_cm,
                color_id=self.color_id,
                light=self.light,
                bumpers=self.bumpers,
                battery_mv=self.battery_mv,
                flags=flags,
            )

    def _send(self, frame: bytes) -> bool:
        try:
            self._sock.sendall(frame)
            return True
        except OSError:
            return False


def _normalise(degrees: float) -> float:
    """Wrap into [-180, 180), matching Protocol.normaliseDegrees in Java."""
    while degrees >= 180.0:
        degrees -= 360.0
    while degrees < -180.0:
        degrees += 360.0
    return degrees


def make_simulated_pair(
    telemetry_period_ms: int = 50, world: "SimulatedRoom | None" = None
):
    """Return ``(pi_socket, simulator)`` joined by an in-process socket pair."""
    pi_side, nxt_side = socket.socketpair()
    sim = SimulatedNXT(nxt_side, telemetry_period_ms=telemetry_period_ms, world=world)
    sim.start()
    return pi_side, sim


def _serve_tcp(port: int) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(1)
    print(f"Simulated NXT listening on 127.0.0.1:{port} (Ctrl-C to stop)")
    while True:
        conn, addr = listener.accept()
        print(f"connected: {addr}")
        sim = SimulatedNXT(conn)
        sim.start()
        try:
            while sim._thread is not None and sim._thread.is_alive():
                time.sleep(0.2)
        except KeyboardInterrupt:
            sim.stop()
            return
        print("disconnected")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulated NXT ScoutServer")
    parser.add_argument("--port", type=int, default=5555)
    _serve_tcp(parser.parse_args().port)
