"""High-level robot API over the wire protocol.

Owns a background reader thread that keeps the latest telemetry available
without the caller having to poll the link. Everything above this layer
(mapping, exploration, the mission state machine) talks to this class rather
than to the protocol or transport directly.

Threading contract
------------------
One reader thread, started by :meth:`connect`. Telemetry is published to
``self.telemetry`` under a lock; event callbacks fire *on the reader thread*,
so keep them short and do not issue blocking robot commands from inside one.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Deque, Optional

from . import protocol as p
from .transport import Transport, TransportError

log = logging.getLogger(__name__)

EventCallback = Callable[[int], None]
TelemetryCallback = Callable[[p.Telemetry], None]


class RobotError(RuntimeError):
    """A command could not be completed."""


class Robot:
    """Client for the NXT's ScoutServer firmware."""

    def __init__(self, transport: Transport, telemetry_period_ms: int = 100) -> None:
        self.transport = transport
        self.telemetry_period_ms = telemetry_period_ms

        self._decoder = p.FrameDecoder()
        self._lock = threading.Lock()
        self._reader: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._telemetry: Optional[p.Telemetry] = None
        self._telemetry_time: float = 0.0
        self._telemetry_seen = threading.Event()

        self._pong = threading.Event()
        self._move_done = threading.Event()
        self._turret_done = threading.Event()
        self._last_ack: Optional[tuple[int, int]] = None
        self._ack_received = threading.Event()

        self._event_callbacks: list[EventCallback] = []
        self._telemetry_callbacks: list[TelemetryCallback] = []

        #: Recent events, for the dashboard and the voice agent's context.
        self.events: Deque[tuple[float, int]] = deque(maxlen=64)

        self.frames_received = 0

    # ---------------------------------------------------------------- lifecycle

    def connect(self, timeout: float = 5.0) -> None:
        """Open the link, start the reader, and verify the firmware answers."""
        self.transport.open()
        self._stop.clear()
        self._reader = threading.Thread(
            target=self._read_loop, name="scout-reader", daemon=True
        )
        self._reader.start()

        if not self.ping(timeout=timeout):
            self.close()
            raise RobotError(
                "no PONG from the NXT. Check that ScoutServer is running on the "
                "brick and that it is waiting for a Bluetooth connection."
            )
        self.set_telemetry_period(self.telemetry_period_ms)

    def close(self) -> None:
        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=2.0)
            self._reader = None
        self.transport.close()

    def __enter__(self) -> "Robot":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        # Stop the motors before dropping the link. The firmware would not
        # otherwise know the mission ended, and would happily keep driving.
        try:
            self.stop()
        except (RobotError, TransportError):
            pass
        self.close()

    # ------------------------------------------------------------------ reading

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self.transport.read(timeout=0.1)
            except TransportError as exc:
                log.error("link lost: %s", exc)
                self._stop.set()
                return
            if not chunk:
                continue
            for frame_type, payload in self._decoder.feed(chunk):
                self.frames_received += 1
                self._handle(frame_type, payload)

    def _handle(self, frame_type: int, payload: bytes) -> None:
        if frame_type == p.RSP_TELEMETRY:
            try:
                telemetry = p.Telemetry.decode(payload)
            except p.ProtocolError as exc:
                log.warning("bad telemetry frame: %s", exc)
                return
            with self._lock:
                self._telemetry = telemetry
                self._telemetry_time = time.monotonic()
            self._telemetry_seen.set()
            for callback in list(self._telemetry_callbacks):
                self._safe_call(callback, telemetry)

        elif frame_type == p.RSP_PONG:
            self._pong.set()

        elif frame_type == p.RSP_EVENT and payload:
            code = payload[0]
            self.events.append((time.time(), code))
            if code == p.EV_MOVE_DONE:
                self._move_done.set()
            elif code == p.EV_TURRET_DONE:
                self._turret_done.set()
            log.info("event: %s", p.EVENT_NAMES.get(code, f"0x{code:02X}"))
            for callback in list(self._event_callbacks):
                self._safe_call(callback, code)

        elif frame_type == p.RSP_ACK and len(payload) == 2:
            self._last_ack = (payload[0], payload[1])
            self._ack_received.set()

        elif frame_type == p.RSP_LOG:
            log.info("nxt: %s", payload.decode("ascii", "replace"))

    @staticmethod
    def _safe_call(callback, argument) -> None:
        """Never let a misbehaving callback kill the reader thread."""
        try:
            callback(argument)
        except Exception:  # noqa: BLE001
            log.exception("callback raised; continuing")

    # ------------------------------------------------------------------ writing

    def _send(self, frame: bytes) -> None:
        if self._stop.is_set():
            raise RobotError("the link is down")
        try:
            self.transport.write(frame)
        except TransportError as exc:
            raise RobotError(f"send failed: {exc}") from exc

    def _send_awaiting_ack(self, frame: bytes, timeout: float = 1.0) -> int:
        """Send and wait for the firmware's ACK, returning its status code."""
        self._ack_received.clear()
        self._send(frame)
        if not self._ack_received.wait(timeout):
            raise RobotError("timed out waiting for an ACK from the NXT")
        assert self._last_ack is not None
        return self._last_ack[1]

    # ----------------------------------------------------------------- commands

    def ping(self, timeout: float = 2.0) -> bool:
        self._pong.clear()
        self._send(p.ping())
        return self._pong.wait(timeout)

    def stop(self) -> None:
        self._send_awaiting_ack(p.stop())

    def drive(self, v_mm_s: float, omega_deg_s: float = 0.0) -> bool:
        """Continuous velocity control, for teleop.

        Returns False if the firmware refused because its local safety layer
        is currently blocking forward motion.
        """
        status = self._send_awaiting_ack(p.drive(v_mm_s, omega_deg_s))
        return status == p.ACK_OK

    def travel(self, distance_mm: float, wait: bool = True, timeout: float = 60.0) -> bool:
        """Drive straight. Returns False if refused or if it did not finish."""
        self._move_done.clear()
        if self._send_awaiting_ack(p.travel(distance_mm)) != p.ACK_OK:
            return False
        return self._move_done.wait(timeout) if wait else True

    def turn_to(self, heading_deg: float, wait: bool = True, timeout: float = 30.0) -> bool:
        """Rotate to an absolute heading in the odometry frame."""
        self._move_done.clear()
        if self._send_awaiting_ack(p.turn_to(heading_deg)) != p.ACK_OK:
            return False
        return self._move_done.wait(timeout) if wait else True

    def turret_to(self, angle_deg: float, wait: bool = True, timeout: float = 10.0) -> bool:
        """Point the scanning head.

        The firmware clamps to its travel limit to protect the camera ribbon,
        so a request beyond the limit succeeds but lands short. Read the actual
        bearing back from telemetry rather than assuming the request was met.
        """
        self._send_awaiting_ack(p.turret_to(angle_deg))
        if not wait:
            return True
        # Captured AFTER the ACK returns, not before sending: the ACK is the
        # firmware's guarantee that it has already applied this command
        # (updated its target), so any telemetry frame it generates from
        # this point on reflects the new target. Capturing the baseline
        # before sending is NOT enough — a telemetry frame can arrive after
        # that earlier timestamp while still describing state from BEFORE
        # the firmware processed our command, purely due to network/
        # scheduling latency between "we sent it" and "the firmware acted
        # on it." Caught via live testing: with the earlier (too-early)
        # baseline, turret_to(60) could still return true while showing the
        # tail end of the *previous* command's motion, because a stale-ish
        # frame slipped in just after the naive baseline but before the
        # firmware had actually started the new move.
        baseline = time.monotonic()
        return self._wait_until(
            lambda t: not (t.flags & p.FLAG_TURRET_MOVING), timeout=timeout, after=baseline
        )

    def set_pose(self, x_mm: float, y_mm: float, heading_deg: float) -> None:
        """Overwrite the firmware's odometry, e.g. after an ArUco fix."""
        self._send_awaiting_ack(p.set_pose(x_mm, y_mm, heading_deg))

    def set_safety(self, enabled: bool, min_range_cm: int = 20) -> None:
        """Configure the firmware's local safety stop.

        Disabling this does **not** disable the bumpers — those always stop the
        robot. It only relaxes the ultrasonic threshold, which is occasionally
        useful for deliberate close approaches.
        """
        self._send_awaiting_ack(p.set_safety(enabled, min_range_cm))

    def set_telemetry_period(self, period_ms: int) -> None:
        self._send_awaiting_ack(p.set_telemetry(period_ms))
        self.telemetry_period_ms = period_ms

    # -------------------------------------------------------------------- state

    @property
    def telemetry(self) -> Optional[p.Telemetry]:
        """The most recent telemetry frame, or None if none has arrived."""
        with self._lock:
            return self._telemetry

    @property
    def telemetry_age(self) -> float:
        """Seconds since the last telemetry frame; ``inf`` if none yet.

        A value that keeps growing past a few multiples of the telemetry period
        means the link has stalled even though it has not formally dropped.
        """
        with self._lock:
            if self._telemetry is None:
                return float("inf")
            return time.monotonic() - self._telemetry_time

    @property
    def connected(self) -> bool:
        return not self._stop.is_set() and self.transport.is_open

    def wait_for_telemetry(self, timeout: float = 2.0) -> Optional[p.Telemetry]:
        self._telemetry_seen.wait(timeout)
        return self.telemetry

    def _wait_until(
        self,
        predicate: Callable[[p.Telemetry], bool],
        timeout: float,
        after: Optional[float] = None,
    ) -> bool:
        """Poll telemetry until ``predicate`` holds or the timeout expires.

        If ``after`` is given (a `time.monotonic()` timestamp), frames that
        arrived at or before it are ignored — otherwise a frame cached from
        before the caller's command was even sent could satisfy the
        predicate by coincidence, returning success without having actually
        waited for anything.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                telemetry = self._telemetry
                telemetry_time = self._telemetry_time
            if (
                telemetry is not None
                and (after is None or telemetry_time > after)
                and predicate(telemetry)
            ):
                return True
            if self._stop.is_set():
                return False
            time.sleep(0.02)
        return False

    # ---------------------------------------------------------------- callbacks

    def on_event(self, callback: EventCallback) -> None:
        """Register an event callback. Runs on the reader thread — keep it short."""
        self._event_callbacks.append(callback)

    def on_telemetry(self, callback: TelemetryCallback) -> None:
        """Register a telemetry callback. Runs on the reader thread."""
        self._telemetry_callbacks.append(callback)

    def remove_event_callback(self, callback: EventCallback) -> None:
        """Undo a prior `on_event`. A no-op if `callback` isn't registered.

        Matters for any caller whose lifetime is shorter than the Robot's
        own — e.g. ExplorationMission, which is recreated on every
        SCOUT_LOOP restart while `robot` lives for the whole process.
        Without this, each restart's bound `_on_robot_event` method stayed
        in `_event_callbacks` forever, keeping that lap's whole mission
        object — grid, point cloud, everything — permanently unreachable
        but un-freeable. Confirmed as the cause of a real OOM kill on the
        public demo (Render reported "exited with status 137" /
        "exceeded its memory limit" after enough SCOUT_LOOP laps).
        """
        try:
            self._event_callbacks.remove(callback)
        except ValueError:
            pass

    # -------------------------------------------------------------- diagnostics

    def link_stats(self) -> dict:
        """Link health, for the dashboard.

        ``checksum_errors`` climbing steadily is the signature of the
        WiFi/Bluetooth coexistence problem — see transport.py.
        """
        return {
            "frames_received": self.frames_received,
            "checksum_errors": self._decoder.checksum_errors,
            "resyncs": self._decoder.resyncs,
            "telemetry_age_s": round(self.telemetry_age, 3),
            "connected": self.connected,
        }
