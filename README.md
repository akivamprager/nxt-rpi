# Scout

A LEGO Mindstorms NXT 2.0 robot that explores a room on its own and builds a
live map you watch in a browser.

The design principle is a clean split of responsibilities:

- **The NXT is the body.** A dumb, stable real-time layer in leJOS Java: motor
  control, sensor sampling, and a safety stop that works whether or not the Pi
  is talking to it.
- **The Pi is the brain.** Vision, localisation, mapping, planning, and the IoT
  layer, in Python.

The third motor drives a single turret carrying **both** the ultrasonic sensor
and the Pi camera — one motor buys a scanning rangefinder and a steerable
camera. Printed ArUco markers act as surveyed landmarks so the map does not
smear as wheel odometry drifts.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Toolchain + firmware flash | Needs hardware — see [docs/SETUP.md](docs/SETUP.md) |
| 1 | NXT firmware (`ScoutServer`) | Written, cross-checked against Python byte-for-byte, not yet run on a brick |
| 2 | Transport, protocol, robot API | **Done and tested** |
| 3 | Camera calibration + ArUco pose + localization math | **Math and detection done and tested**; real capture from a Pi Camera Module can't be verified until one exists |
| 4 | Occupancy grid + frontier exploration + mission state machine | **Done, tested, and demoed live end-to-end** |
| 5 | Full MQTT + Flask dashboard + GPIO LEDs | Not started; a zero-dependency stepping-stone dashboard exists (see below) |
| 6 | Voice agent (runs on the Mac) | Not started |

110 tests pass across the Pi stack (`pi/tests/`), none requiring hardware.

### Phase 3 in detail — what's tested vs. what's genuinely waiting on hardware

Localization splits cleanly into two layers, and only one of them needs a
camera:

- **`pose2d.py` + `localize.py`** — the 2D rigid-transform math that turns
  "a marker's pose relative to the camera" into "the robot's corrected world
  pose." Pure geometry, no opencv. Fully tested with hand-derived and
  round-trip cases.
- **`vision.py`** — ArUco detection and the `(rvec, tvec)` → `Pose2D`
  conversion. This needed real `opencv`/`numpy`, and once installed, every
  sign and axis convention was derived empirically against actual
  `cv2.projectPoints`/`solvePnP` calls (not assumed from documentation) and
  proven by rendering synthetic marker images end-to-end through real
  detection and the tested `localize.py` chain — see `test_vision.py`.
  Camera calibration (`calibrate_from_chessboard_images`) is written
  correctly per the standard OpenCV recipe but is unverified, since no
  camera exists yet to photograph a real checkerboard with.

`mission.py`'s sweep step already has the hook wired in
(`ExplorationMission(..., localizer=...)`) and is tested against a fake
localizer standing in for real vision — see `test_sweep_applies_a_
localizer_correction`. Once a camera exists, a real `MarkerLocalizer`
(vision.py + localize.py + config.yaml's marker poses) plugs into that exact
same hook with no changes to mission.py.

`pi/config.yaml` is the template for everything that's specific to this
physical build — wheel geometry, camera/turret mount offsets, grid bounds,
and surveyed marker positions. Its validation logic (`config.py`) is fully
tested against plain dicts; only the thin "read the YAML file" wrapper needs
PyYAML and is unverified until BUILD.md's measurements exist to fill in.

## Watch it explore a room right now — no hardware at all

```bash
python3 pi/tools/demo_explore.py
```

Open the printed URL. This runs a simulated NXT in a simulated room, driven by
the *actual* Robot/mission/mapping code that will run against real hardware
unchanged — sweeping, mapping, planning, and driving until the room is
explored, live in your browser. Standard library only, nothing to install.

This demo is also how three real bugs got caught before ever touching
hardware: a single-ray ultrasonic model left angular gaps the exploration
loop would get permanently stuck in, and a stuck-target fix could itself get
defeated when only one unreachable frontier remained. All are now covered by
regression tests in `pi/tests/test_mapping.py` and `pi/tests/test_mission.py`.

A fourth, more serious bug surfaced while wiring Phase 3's localization hook
into the mission's sweep step: `Robot.turret_to(wait=True)` could return
success by reading a **stale telemetry frame that didn't yet reflect the
new command** — if the turret happened to already be stationary, an early
poll could see `FLAG_TURRET_MOVING` clear for the *old* reason (hadn't
started) rather than the new one (already arrived), and return immediately
without the turret having moved at all. Reproduced deliberately against the
pre-fix code: **16 of 20 consecutive commands hit the race** — this wasn't
a rare edge case, it was the dominant behavior, just masked in earlier
tests that happened to double-check completion themselves rather than
trusting `turret_to`'s return value.

The first fix attempt (reject any telemetry older than a timestamp captured
*before sending* the command) turned out to be insufficient and still
failed occasionally under stress testing — a telemetry frame can arrive
after that early timestamp due to network/scheduling latency while still
describing state from before the firmware actually processed the command.
The correct fix captures the baseline *after* the command's ACK is
received: the ACK is the firmware's guarantee that it already applied the
command, and the transport's ordered byte stream guarantees no telemetry
reflecting the old state can arrive after it. Verified with 60 rapid
consecutive commands and zero failures; see
`test_turret_to_does_not_return_before_the_turret_actually_moves` in
`pi/tests/test_robot.py`.

## Try it without hardware

The whole Pi stack runs against a simulated brick.

```bash
for f in pi/tests/test_*.py; do python3 "$f" || break; done
```

`test_vision.py` needs `numpy` and `opencv-contrib-python-headless` installed
(`pip install --user numpy opencv-contrib-python-headless pyyaml` — the
headless variant since nothing here ever opens a display window, on this Mac
or the eventual Pi); every other file is standard library only.

To drive a simulated robot by hand, run `python3 pi/tools/sim_firmware.py --port 5555`
in one terminal and `python3 pi/tools/teleop.py --sim` in another.

## Layout

```
nxt-firmware/       leJOS Java — built and uploaded from the Windows laptop
  src/scout/        Protocol, Turret, ScoutServer
  test/             ProtocolTest + generated golden vectors
pi/                 Python — runs on the Raspberry Pi
  config.yaml        This build's measurements: geometry, mounts, grid, markers
  scout/            protocol, transport, robot, mapping, explore, mission,
                     pose2d, localize, vision, config
  tools/            teleop, firmware + room simulator, demo_explore, vector generator
  tests/            one test file per module above, all hardware-free
  web/              zero-dependency live dashboard (server.py + index.html)
docs/SETUP.md       Toolchain, pairing, and troubleshooting
docs/BUILD.md       Physical LEGO build, against the real Quick Start instructions
```

## Two things that will bite you

**The protocol lives in two files that must agree.** `pi/scout/protocol.py` and
`nxt-firmware/src/scout/Protocol.java` are mirrors. Change one, change the
other, then regenerate and re-run the cross-check:

```bash
python3 pi/tools/gen_vectors.py
```

A mismatch does not announce itself — it looks like a robot that silently
ignores commands.

**The turret travel limit is hardware protection, not a preference.** The
camera's CSI ribbon runs from the rotating head down to the Pi. Continuous
rotation will tear it. `TURRET_MAX_ANGLE_DEG` in `ScoutServer.java` and the
clamp in `Turret.java` exist for that reason; lower it if your build needs it.

## Design notes

Full reasoning, hardware decisions, and the phase plan live in the plan file at
`~/.claude/plans/ive-got-a-mindstorms-quiet-dewdrop.md`. The short version of
the two decisions that shaped the most code:

- **Bluetooth RFCOMM, not USB.** `/dev/rfcomm0` is a file you open; the USB
  path is hours of undocumented leJOS handshake work. `transport.py` abstracts
  it so switching is small if it ever becomes necessary.
- **Video never goes over WiFi raw.** The Pi 3B's WiFi and Bluetooth are the
  same chip sharing one antenna, so saturating WiFi stutters the robot link.
  Detection happens on the Pi; only low-rate frames ship out.
