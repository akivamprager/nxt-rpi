"""Round-trip and robustness tests for the wire protocol.

These run without any hardware, which is the point: the framing layer is the
one part of the NXT link that can be fully verified from a laptop. Everything
above it needs a brick on the other end.

Run with:  python3 -m pytest pi/tests/ -q
"""

import os
import random
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scout import protocol as p  # noqa: E402


def decode_all(data: bytes):
    return p.FrameDecoder().feed(data)


def test_encode_decode_round_trip():
    frames = decode_all(p.encode(p.CMD_PING, b"\x01\x02\x03"))
    assert frames == [(p.CMD_PING, b"\x01\x02\x03")]


def test_empty_payload():
    assert decode_all(p.ping()) == [(p.CMD_PING, b"")]


def test_max_payload_is_accepted():
    payload = bytes(range(256))[:255]
    assert decode_all(p.encode(p.RSP_LOG, payload)) == [(p.RSP_LOG, payload)]


def test_oversized_payload_is_rejected():
    try:
        p.encode(p.RSP_LOG, b"\x00" * 256)
    except p.ProtocolError:
        return
    raise AssertionError("expected ProtocolError for a 256-byte payload")


def test_command_builders_round_trip():
    cases = [
        (p.drive(250, -45.0), p.CMD_DRIVE, ">hh", (250, -450)),
        (p.travel(-1234), p.CMD_TRAVEL, ">i", (-1234,)),
        (p.turn_to(90.0), p.CMD_TURN_TO, ">h", (900,)),
        (p.turret_to(-120.0), p.CMD_TURRET_TO, ">h", (-1200,)),
        (p.set_pose(100, -200, 45.0), p.CMD_SET_POSE, ">iih", (100, -200, 450)),
        (p.set_safety(True, 20), p.CMD_SET_SAFETY, ">BB", (1, 20)),
        (p.set_telemetry(100), p.CMD_SET_TELEMETRY, ">H", (100,)),
    ]
    for wire, expected_type, fmt, expected_values in cases:
        (frame_type, payload), = decode_all(wire)
        assert frame_type == expected_type
        assert struct.unpack(fmt, payload) == expected_values


def test_telemetry_round_trip():
    original = p.Telemetry(
        seq=65535,
        x_mm=-1500.0,
        y_mm=2400.0,
        heading_deg=-179.9,
        turret_deg=120.0,
        range_cm=42,
        color_id=3,
        light=77,
        bumpers=0b10,
        battery_mv=8123,
        flags=p.FLAG_MOVING | p.FLAG_SAFETY_ENABLED,
    )
    restored = p.Telemetry.decode(original.encode())
    assert restored == original
    assert restored.moving is True
    assert restored.safety_tripped is False
    assert restored.bumper_pressed is True
    assert restored.has_echo is True


def test_no_echo_is_distinguishable_from_close_range():
    """255 means 'no information', and must not read as an obstacle or as clear."""
    far = p.Telemetry.decode(
        p.Telemetry(0, 0, 0, 0, 0, p.US_NO_ECHO, 0, 0, 0, 8000, 0).encode()
    )
    near = p.Telemetry.decode(p.Telemetry(0, 0, 0, 0, 0, 10, 0, 0, 0, 8000, 0).encode())
    assert far.has_echo is False
    assert near.has_echo is True


def test_streaming_split_across_arbitrary_chunks():
    """A frame split across reads must still decode. RFCOMM will do this."""
    wire = p.encode(p.RSP_TELEMETRY, b"\xAA" * 20)
    for split in range(1, len(wire)):
        decoder = p.FrameDecoder()
        got = decoder.feed(wire[:split]) + decoder.feed(wire[split:])
        assert got == [(p.RSP_TELEMETRY, b"\xAA" * 20)], f"failed at split {split}"


def test_byte_at_a_time_delivery():
    wire = p.encode(p.CMD_DRIVE, b"\x01\x02\x03\x04")
    decoder = p.FrameDecoder()
    frames = []
    for byte in wire:
        frames.extend(decoder.feed(bytes([byte])))
    assert frames == [(p.CMD_DRIVE, b"\x01\x02\x03\x04")]


def test_back_to_back_frames_in_one_read():
    wire = p.ping() + p.stop() + p.travel(500)
    types = [t for t, _ in decode_all(wire)]
    assert types == [p.CMD_PING, p.CMD_STOP, p.CMD_TRAVEL]


def test_leading_garbage_is_discarded():
    decoder = p.FrameDecoder()
    frames = decoder.feed(b"\x00\xFF\xA5garbage" + p.ping())
    assert frames == [(p.CMD_PING, b"")]
    assert decoder.resyncs > 0


def test_corrupted_checksum_drops_only_that_frame():
    good = p.encode(p.CMD_TRAVEL, b"\x00\x00\x01\x00")
    bad = bytearray(good)
    bad[-1] ^= 0xFF
    decoder = p.FrameDecoder()
    frames = decoder.feed(bytes(bad) + p.ping())
    assert frames == [(p.CMD_PING, b"")], "the following good frame must survive"
    assert decoder.checksum_errors == 1


def test_truncated_frame_does_not_wedge_the_decoder():
    """A dropped tail must not stall the stream forever.

    The truncated frame claims a 30-byte payload (35 bytes total) but only 10
    arrive. The decoder must keep waiting, then — once enough bytes exist to
    evaluate it — fail the checksum and resync rather than blocking.
    """
    truncated = p.encode(p.RSP_TELEMETRY, b"\x01" * 30)[:10]
    decoder = p.FrameDecoder()
    assert decoder.feed(truncated) == []
    assert decoder.feed(p.ping()) == []  # 15 bytes: still short of the claimed 35

    # Six more pings push the buffer past 35 bytes, so the bad frame is finally
    # evaluated and rejected.
    frames = decoder.feed(p.ping() * 6)
    assert decoder.checksum_errors == 1
    assert frames, "decoder must not stay wedged once the bad frame is resolved"


def test_good_frames_swallowed_by_a_bad_length_are_recovered():
    """A corrupt length byte must not take the frames behind it down with it.

    This is why a failed checksum drops only the sync word rather than the
    whole claimed frame: the 'payload' it swallowed is usually real data.
    """
    truncated = p.encode(p.RSP_TELEMETRY, b"\x01" * 30)[:10]
    decoder = p.FrameDecoder()
    decoder.feed(truncated)
    frames = decoder.feed(p.ping() * 6)
    recovered = [f for f in frames if f == (p.CMD_PING, b"")]
    assert len(recovered) >= 5, f"only recovered {len(recovered)} of 6 buried frames"


def test_fuzz_recovers_and_never_raises():
    """Random noise must never crash the decoder, and real frames must survive it."""
    rng = random.Random(1234)
    decoder = p.FrameDecoder()
    recovered = 0
    for _ in range(300):
        noise = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 12)))
        decoder.feed(noise)
        for frame_type, payload in decoder.feed(p.turn_to(90.0)):
            if frame_type == p.CMD_TURN_TO and payload == struct.pack(">h", 900):
                recovered += 1
    # Noise can legitimately swallow some frames by forging a sync word and a
    # plausible length. The decoder must recover promptly, not perfectly.
    assert recovered > 200, f"only recovered {recovered}/300 frames after noise"


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a deliberately tiny test runner
            print(f"FAIL {name}: {exc}")
            failed += 1
        else:
            print(f"ok   {name}")
            passed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
