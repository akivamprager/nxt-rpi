"""Scout wire protocol — Pi side.

This module MUST stay byte-for-byte in sync with
``nxt-firmware/src/scout/Protocol.java``. If you change an opcode or a
payload layout here, change it there in the same commit.

Framing
-------
We open the Bluetooth link in ``NXTConnection.RAW`` mode, which means leJOS
does no framing for us and we see a plain byte stream. So we frame it
ourselves::

    A5 5A  LEN  TYPE  PAYLOAD[LEN]  XOR

- ``A5 5A`` is a sync word, so a receiver that starts mid-stream (or drops
  bytes) can hunt forward and re-lock instead of being desynced forever.
- ``LEN`` is the payload length only (0-255), excluding TYPE and XOR.
- ``XOR`` is TYPE xor'd with every payload byte. This is a corruption check,
  not a security measure.

All multi-byte integers are **big-endian**, because that is what Java's
``DataOutputStream``/``DataInputStream`` do natively and matching them keeps
the firmware side free of manual byte swapping.

Units
-----
Fixed-point integers throughout; the NXT has no FPU worth using and integers
keep the frames small and unambiguous.

- distances: millimetres (int32)
- angles: deci-degrees, i.e. 1/10 deg (int16), so 900 == 90.0 deg
- speeds: mm/s (int16) and deci-deg/s (int16)
- ultrasonic range: centimetres (uint8), 255 == no echo
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

SYNC0 = 0xA5
SYNC1 = 0x5A

MAX_PAYLOAD = 255
#: Sync(2) + LEN(1) + TYPE(1) + payload + XOR(1)
FRAME_OVERHEAD = 5

# --------------------------------------------------------------------------
# Opcodes: Pi -> NXT (commands). High bit clear.
# --------------------------------------------------------------------------
CMD_PING = 0x01
CMD_DRIVE = 0x02  # int16 v_mm_s, int16 omega_ddeg_s
CMD_TRAVEL = 0x03  # int32 distance_mm
CMD_TURN_TO = 0x04  # int16 heading_ddeg (absolute, odometry frame)
CMD_STOP = 0x05
CMD_TURRET_TO = 0x06  # int16 angle_ddeg (relative to turret zero)
CMD_SET_POSE = 0x07  # int32 x_mm, int32 y_mm, int16 heading_ddeg
CMD_SET_SAFETY = 0x08  # uint8 enabled, uint8 min_range_cm
CMD_SET_TELEMETRY = 0x09  # uint16 period_ms (0 disables)

# --------------------------------------------------------------------------
# Opcodes: NXT -> Pi (responses). High bit set, so a stray byte is easy to
# classify as "wrong direction" during debugging.
# --------------------------------------------------------------------------
RSP_PONG = 0x81
RSP_TELEMETRY = 0x82
RSP_EVENT = 0x83
RSP_ACK = 0x84
RSP_LOG = 0x85

# --------------------------------------------------------------------------
# Event codes carried by RSP_EVENT.
# --------------------------------------------------------------------------
EV_BUMPER = 0x01  # a touch sensor closed
EV_SAFETY_STOP = 0x02  # ultrasonic tripped the local safety threshold
EV_MOVE_DONE = 0x03  # a TRAVEL / TURN_TO completed
EV_TURRET_DONE = 0x04
EV_STALL = 0x05
EV_LOW_BATTERY = 0x06

EVENT_NAMES = {
    EV_BUMPER: "bumper",
    EV_SAFETY_STOP: "safety_stop",
    EV_MOVE_DONE: "move_done",
    EV_TURRET_DONE: "turret_done",
    EV_STALL: "stall",
    EV_LOW_BATTERY: "low_battery",
}

# ACK status codes.
ACK_OK = 0x00
ACK_BAD_LENGTH = 0x01
ACK_UNKNOWN_CMD = 0x02
ACK_REFUSED = 0x03  # e.g. motion refused because a bumper is held closed

#: Ultrasonic sentinel: leJOS returns 255 from getDistance() when no echo
#: came back. That means "I learned nothing", NOT "the way is clear" — the
#: mapping code must treat it as unknown, not as free space.
US_NO_ECHO = 255

#: Telemetry payload: seq, x, y, heading, turret, range, colour, light,
#: bumpers, battery, flags.
_TELEMETRY_FMT = ">HiihhBBBBHB"
TELEMETRY_SIZE = struct.calcsize(_TELEMETRY_FMT)

# Bit masks for the telemetry ``flags`` byte.
FLAG_MOVING = 0x01
FLAG_TURRET_MOVING = 0x02
FLAG_SAFETY_TRIPPED = 0x04
FLAG_SAFETY_ENABLED = 0x08


class ProtocolError(Exception):
    """Raised on a malformed frame that a caller could reasonably recover from."""


#: Internal sentinels for FrameDecoder._step. They must be distinguishable
#: from a decoded ``(type, payload)`` tuple and from each other.
_NEED_MORE = object()
_DROPPED = object()


def checksum(frame_type: int, payload: bytes) -> int:
    """XOR of the type byte and every payload byte."""
    value = frame_type & 0xFF
    for byte in payload:
        value ^= byte
    return value & 0xFF


def encode(frame_type: int, payload: bytes = b"") -> bytes:
    """Wrap ``payload`` in a framed, checksummed packet ready for the wire."""
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(
            f"payload of {len(payload)} bytes exceeds the {MAX_PAYLOAD}-byte "
            "limit imposed by the single-byte length field"
        )
    return bytes(
        [SYNC0, SYNC1, len(payload), frame_type & 0xFF]
        + list(payload)
        + [checksum(frame_type, payload)]
    )


@dataclass(frozen=True)
class Telemetry:
    """One decoded telemetry frame.

    Fields are converted to human units here so nothing downstream has to
    remember the fixed-point scaling.
    """

    seq: int
    x_mm: float
    y_mm: float
    heading_deg: float
    turret_deg: float
    range_cm: int
    color_id: int
    light: int
    bumpers: int
    battery_mv: int
    flags: int

    @property
    def has_echo(self) -> bool:
        """False when the ultrasonic returned no echo (unknown, not clear)."""
        return self.range_cm != US_NO_ECHO

    @property
    def moving(self) -> bool:
        return bool(self.flags & FLAG_MOVING)

    @property
    def safety_tripped(self) -> bool:
        return bool(self.flags & FLAG_SAFETY_TRIPPED)

    @property
    def bumper_pressed(self) -> bool:
        return self.bumpers != 0

    @classmethod
    def decode(cls, payload: bytes) -> "Telemetry":
        if len(payload) != TELEMETRY_SIZE:
            raise ProtocolError(
                f"telemetry payload is {len(payload)} bytes, expected {TELEMETRY_SIZE}"
            )
        (
            seq,
            x,
            y,
            heading,
            turret,
            rng,
            color,
            light,
            bumpers,
            battery,
            flags,
        ) = struct.unpack(_TELEMETRY_FMT, payload)
        return cls(
            seq=seq,
            x_mm=float(x),
            y_mm=float(y),
            heading_deg=heading / 10.0,
            turret_deg=turret / 10.0,
            range_cm=rng,
            color_id=color,
            light=light,
            bumpers=bumpers,
            battery_mv=battery,
            flags=flags,
        )

    def encode(self) -> bytes:
        """Re-encode. Used by the firmware simulator and the round-trip tests."""
        return struct.pack(
            _TELEMETRY_FMT,
            self.seq,
            int(round(self.x_mm)),
            int(round(self.y_mm)),
            int(round(self.heading_deg * 10)),
            int(round(self.turret_deg * 10)),
            self.range_cm,
            self.color_id,
            self.light,
            self.bumpers,
            self.battery_mv,
            self.flags,
        )


# --------------------------------------------------------------------------
# Command builders. Keeping these as functions (rather than scattering
# struct.pack calls through robot.py) means the wire layout is described in
# exactly one place per command.
# --------------------------------------------------------------------------


def ping() -> bytes:
    return encode(CMD_PING)


def stop() -> bytes:
    return encode(CMD_STOP)


def drive(v_mm_s: float, omega_deg_s: float) -> bytes:
    payload = struct.pack(">hh", int(round(v_mm_s)), int(round(omega_deg_s * 10)))
    return encode(CMD_DRIVE, payload)


def travel(distance_mm: float) -> bytes:
    return encode(CMD_TRAVEL, struct.pack(">i", int(round(distance_mm))))


def turn_to(heading_deg: float) -> bytes:
    return encode(CMD_TURN_TO, struct.pack(">h", int(round(heading_deg * 10))))


def turret_to(angle_deg: float) -> bytes:
    return encode(CMD_TURRET_TO, struct.pack(">h", int(round(angle_deg * 10))))


def set_pose(x_mm: float, y_mm: float, heading_deg: float) -> bytes:
    payload = struct.pack(
        ">iih", int(round(x_mm)), int(round(y_mm)), int(round(heading_deg * 10))
    )
    return encode(CMD_SET_POSE, payload)


def set_safety(enabled: bool, min_range_cm: int) -> bytes:
    return encode(
        CMD_SET_SAFETY, struct.pack(">BB", 1 if enabled else 0, min_range_cm & 0xFF)
    )


def set_telemetry(period_ms: int) -> bytes:
    return encode(CMD_SET_TELEMETRY, struct.pack(">H", period_ms))


class FrameDecoder:
    """Incremental frame decoder for a lossy byte stream.

    Feed it whatever bytes arrive; it yields complete, checksum-verified
    frames as ``(type, payload)``. It is deliberately tolerant: garbage
    between frames is discarded while hunting for the sync word, and a bad
    checksum drops only that frame rather than the whole stream.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        #: Count of frames dropped for a bad checksum. Worth surfacing on the
        #: dashboard: a steadily climbing value is the signature of the
        #: WiFi/Bluetooth coexistence problem described in the plan.
        self.checksum_errors = 0
        self.resyncs = 0

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        """Add received bytes, returning any frames that are now complete."""
        self._buf.extend(data)
        frames: list[tuple[int, bytes]] = []
        while True:
            result = self._step()
            if result is _NEED_MORE:
                return frames
            if result is _DROPPED:
                # Keep going: there may be more complete frames already
                # buffered behind the one we just discarded.
                continue
            frames.append(result)  # type: ignore[arg-type]

    def _step(self):
        """Try to consume one frame. Returns a frame, ``_DROPPED``, or ``_NEED_MORE``."""
        buf = self._buf

        # Hunt for the sync word, discarding anything before it.
        limit = len(buf) - 1
        start = 0
        while start < limit and not (buf[start] == SYNC0 and buf[start + 1] == SYNC1):
            start += 1

        if start >= limit:
            # No sync word in what we hold. Keep at most one trailing byte: it
            # could be the first half of a sync word still in flight.
            if len(buf) > 1:
                del buf[: len(buf) - 1]
                self.resyncs += 1
            return _NEED_MORE

        if start:
            del buf[:start]
            self.resyncs += 1

        if len(buf) < FRAME_OVERHEAD:
            return _NEED_MORE

        length = buf[2]
        total = FRAME_OVERHEAD + length
        if len(buf) < total:
            return _NEED_MORE

        frame_type = buf[3]
        payload = bytes(buf[4 : 4 + length])

        if buf[4 + length] != checksum(frame_type, payload):
            self.checksum_errors += 1
            # Drop only the sync word, not the whole claimed frame. A corrupted
            # length byte can make a short frame claim to be long and swallow
            # good frames behind it; re-hunting from just past the sync word
            # lets those be recovered instead of discarded with it.
            del buf[:2]
            return _DROPPED

        del buf[:total]
        return frame_type, payload
