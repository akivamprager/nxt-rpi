"""Generate golden protocol vectors for the Java cross-check.

The Python protocol implementation is covered by ``pi/tests/test_protocol.py``,
so it is the reference. This script freezes that behaviour into a data file
that ``nxt-firmware/test/ProtocolTest.java`` replays, proving the firmware's
decoder agrees with the Pi's byte for byte.

That agreement is the single highest-risk seam in the project: a mismatch
shows up as a robot that silently ignores commands, which is miserable to
diagnose with a brick on the end of a Bluetooth link.

Regenerate after any protocol change::

    python3 pi/tools/gen_vectors.py
"""

from __future__ import annotations

import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from scout import protocol as p  # noqa: E402

OUTPUT = os.path.join(_HERE, "..", "..", "nxt-firmware", "test", "vectors.txt")


def hexify(data: bytes) -> str:
    return data.hex().upper() if data else "-"


def frame_vectors():
    """(name, type, payload) triples covering every command and response."""
    return [
        ("ping", p.CMD_PING, b""),
        ("stop", p.CMD_STOP, b""),
        ("drive_fwd", p.CMD_DRIVE, struct.pack(">hh", 250, 0)),
        ("drive_arc", p.CMD_DRIVE, struct.pack(">hh", 150, -450)),
        ("drive_spin", p.CMD_DRIVE, struct.pack(">hh", 0, 900)),
        ("drive_reverse", p.CMD_DRIVE, struct.pack(">hh", -200, 0)),
        ("travel_fwd", p.CMD_TRAVEL, struct.pack(">i", 1000)),
        ("travel_back", p.CMD_TRAVEL, struct.pack(">i", -250)),
        ("travel_zero", p.CMD_TRAVEL, struct.pack(">i", 0)),
        ("turn_90", p.CMD_TURN_TO, struct.pack(">h", 900)),
        ("turn_neg179_9", p.CMD_TURN_TO, struct.pack(">h", -1799)),
        ("turret_center", p.CMD_TURRET_TO, struct.pack(">h", 0)),
        ("turret_max", p.CMD_TURRET_TO, struct.pack(">h", 1200)),
        ("turret_beyond", p.CMD_TURRET_TO, struct.pack(">h", 4000)),
        ("set_pose", p.CMD_SET_POSE, struct.pack(">iih", 1234, -5678, 450)),
        ("set_pose_zero", p.CMD_SET_POSE, struct.pack(">iih", 0, 0, 0)),
        ("safety_on", p.CMD_SET_SAFETY, struct.pack(">BB", 1, 20)),
        ("safety_off", p.CMD_SET_SAFETY, struct.pack(">BB", 0, 255)),
        ("telemetry_100", p.CMD_SET_TELEMETRY, struct.pack(">H", 100)),
        ("telemetry_off", p.CMD_SET_TELEMETRY, struct.pack(">H", 0)),
        ("pong", p.RSP_PONG, b""),
        ("ack_ok", p.RSP_ACK, bytes([p.CMD_TRAVEL, p.ACK_OK])),
        ("ack_refused", p.RSP_ACK, bytes([p.CMD_DRIVE, p.ACK_REFUSED])),
        ("event_bumper", p.RSP_EVENT, bytes([p.EV_BUMPER])),
        ("event_safety", p.RSP_EVENT, bytes([p.EV_SAFETY_STOP])),
        ("log", p.RSP_LOG, b"hello"),
        # A payload containing bytes that look like a sync word, to prove the
        # length field (not scanning) is what delimits a frame.
        ("payload_with_sync", p.RSP_LOG, bytes([0xA5, 0x5A, 0xA5, 0x5A])),
        ("payload_max", p.RSP_LOG, bytes(range(255))),
    ]


def telemetry_vectors():
    return [
        p.Telemetry(0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 0),
        p.Telemetry(1, 100, -200, 45.0, 30.0, 55, 3, 128, 0, 8200, p.FLAG_MOVING),
        # Extremes: the widest values each field must survive.
        p.Telemetry(
            65535, 2147483647, -2147483648, 179.9, -120.0, 254, 17, 255, 3, 65535, 0x0F
        ),
        # No echo must round-trip distinctly from a real reading.
        p.Telemetry(7, -1, 1, -179.9, 0.0, p.US_NO_ECHO, 0, 0, 0, 6100, 0),
        p.Telemetry(
            8, 12345, -6789, 90.0, 119.9, 20, 5, 64, 1,
            7400, p.FLAG_SAFETY_TRIPPED | p.FLAG_SAFETY_ENABLED,
        ),
    ]


def stream_vectors():
    """(name, input bytes, expected decoded frames) — decoder robustness."""
    cases = []

    cases.append(("clean_single", p.ping(), [(p.CMD_PING, b"")]))

    back_to_back = p.ping() + p.stop() + p.travel(500)
    cases.append((
        "back_to_back",
        back_to_back,
        [(p.CMD_PING, b""), (p.CMD_STOP, b""),
         (p.CMD_TRAVEL, struct.pack(">i", 500))],
    ))

    cases.append((
        "leading_garbage",
        b"\x00\xFF\xA5\x11garbage" + p.ping(),
        [(p.CMD_PING, b"")],
    ))

    # A lone sync byte before a real frame: the hunt must not lock onto it.
    cases.append(("false_sync_prefix", b"\xA5" + p.stop(), [(p.CMD_STOP, b"")]))

    corrupt = bytearray(p.encode(p.CMD_TRAVEL, struct.pack(">i", 256)))
    corrupt[-1] ^= 0xFF
    cases.append((
        "bad_checksum_then_good",
        bytes(corrupt) + p.ping(),
        [(p.CMD_PING, b"")],
    ))

    # A truncated frame claiming 30 bytes, followed by real frames. Recovering
    # the buried frames is why a failed checksum drops only the sync word.
    truncated = p.encode(p.RSP_TELEMETRY, b"\x01" * 30)[:10]
    cases.append((
        "buried_frames_recovered",
        truncated + p.ping() * 6,
        [(p.CMD_PING, b"")] * 6,
    ))

    cases.append((
        "sync_in_payload",
        p.encode(p.RSP_LOG, bytes([0xA5, 0x5A, 0x00, 0x01])) + p.ping(),
        [(p.RSP_LOG, bytes([0xA5, 0x5A, 0x00, 0x01])), (p.CMD_PING, b"")],
    ))

    return cases


def verify(cases) -> None:
    """Confirm the Python decoder actually produces what we claim it does."""
    for name, data, expected in cases:
        got = p.FrameDecoder().feed(data)
        if got != expected:
            raise AssertionError(
                f"stream vector '{name}' is wrong:\n  expected {expected}\n  got      {got}"
            )


def main() -> None:
    stream_cases = stream_vectors()
    verify(stream_cases)

    lines = [
        "# Scout protocol golden vectors.",
        "# GENERATED by pi/tools/gen_vectors.py - do not edit by hand.",
        "# Replayed by nxt-firmware/test/ProtocolTest.java to prove the Java and",
        "# Python protocol implementations agree byte for byte.",
        "#",
        "# FRAME  <name> <type> <payload|-> <encoded frame>",
        "# TELEM  <seq> <x> <y> <hdg_ddeg> <turret_ddeg> <range> <colour> <light>"
        " <bumpers> <battery> <flags> <payload>",
        "# STREAM <name> <input> <type:payload,...|->",
        "",
    ]

    for name, frame_type, payload in frame_vectors():
        lines.append(
            f"FRAME {name} {frame_type:02X} {hexify(payload)} "
            f"{hexify(p.encode(frame_type, payload))}"
        )

    lines.append("")
    for telemetry in telemetry_vectors():
        lines.append(
            "TELEM {} {} {} {} {} {} {} {} {} {} {} {}".format(
                telemetry.seq,
                int(telemetry.x_mm),
                int(telemetry.y_mm),
                round(telemetry.heading_deg * 10),
                round(telemetry.turret_deg * 10),
                telemetry.range_cm,
                telemetry.color_id,
                telemetry.light,
                telemetry.bumpers,
                telemetry.battery_mv,
                telemetry.flags,
                hexify(telemetry.encode()),
            )
        )

    lines.append("")
    for name, data, expected in stream_cases:
        if expected:
            frames = ",".join(f"{t:02X}:{hexify(pl)}" for t, pl in expected)
        else:
            frames = "-"
        lines.append(f"STREAM {name} {hexify(data)} {frames}")

    lines.append("")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as handle:
        handle.write("\n".join(lines))

    counts = (len(frame_vectors()), len(telemetry_vectors()), len(stream_cases))
    print(f"Wrote {os.path.normpath(OUTPUT)}")
    print(f"  {counts[0]} frame, {counts[1]} telemetry, {counts[2]} stream vectors")


if __name__ == "__main__":
    main()
