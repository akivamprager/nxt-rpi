# Setup

Work through this in order. Phase 0 is the riskiest part of the whole project —
get it working before writing or changing anything else.

You need three machines, each with one job:

| Machine | Job |
|---|---|
| **Windows laptop** | Builds and flashes the NXT firmware. Nothing else. |
| **Raspberry Pi 3B** | Runs the robot brain. Never needs leJOS installed. |
| **Mac** | Edits code, runs the tests and simulator, runs the Phase 6 voice agent. |

The reason for that split: leJOS NXJ 0.9.1 is from February 2012 and ships
prebuilt Windows binaries, so the Windows laptop needs no compiling. Building
it from source on the Pi means fighting 2012-era C against a modern GCC, and
macOS is worse — leJOS's native USB library has no arm64 slice at all.

---

## Phase 0 — Toolchain and firmware flash (Windows laptop)

### Two requirements that cause almost every reported failure

1. **Install a 32-bit JDK 7.** Not 64-bit, not a JRE. `fantom.dll` ships
   32-bit only, so a 64-bit JDK produces the very common
   `NXTCommException: Cannot load NXTComm driver`. Point `JAVA_HOME` at the
   32-bit JDK. A JRE is not enough — you need `javac` to compile.
2. **Install the LEGO Fantom driver *before* connecting the brick.** Version
   1.2.0 or later. Plugging the NXT in first can bind the wrong driver.

### Steps

1. Install the 32-bit JDK 7 and set `JAVA_HOME`.
2. Install the LEGO Fantom driver. Do not connect the NXT yet.
3. Install `leJOS_NXJ_0.9.1beta-3_win32_setup.exe` from
   [SourceForge](https://sourceforge.net/projects/lejos/files/lejos-NXJ/).
   It installs to `C:\Program Files (x86)\leJOS NXJ` and prompts for the JDK path.
4. Add `%NXJ_HOME%\bin` to your `PATH`.
5. Connect the NXT by USB and switch it on.
6. Put it in firmware update mode: hold the reset button (back of the brick,
   upper-left) for **more than 4 seconds**. It will tick audibly. A
   straightened paperclip works well.
7. Run `nxjflash`.

> **This erases the LEGO firmware and everything stored on the brick.** You can
> always restore it later with the official LEGO MINDSTORMS software.

### Two quirks worth knowing before you panic

- If the battery reads empty and the buttons stop responding after flashing,
  **remove and re-seat one battery**. You do not need to flash again.
- **Restart the brick after flashing.** Skipping this causes erratic motor
  behaviour that looks like a wiring fault.

### Verify the protocol implementations agree

Before flashing real firmware, confirm the Java and Python protocol code
produce identical bytes. A mismatch here does not announce itself — it shows
up later as a robot that silently ignores some commands.

From `nxt-firmware/`:

```bash
javac -d build/test src/scout/Protocol.java test/ProtocolTest.java && java -cp build/test ProtocolTest test/vectors.txt
```

Every line must say `ok`. This test is plain Java with no leJOS dependency, so
it runs anywhere a JDK does. (Regenerate the vectors with
`python3 pi/tools/gen_vectors.py` after any protocol change.)

### Build and upload the firmware

From `nxt-firmware/`:

```bash
nxjc -d build -sourcepath src src/scout/ScoutServer.java && nxjlink -o ScoutServer.nxj -cp build scout.ScoutServer && nxjupload ScoutServer.nxj
```

Once the brick is paired, `nxjupload -b ScoutServer.nxj` uploads over
Bluetooth, so you do not need to re-cable for every change.

**Before the first run, measure your robot and edit the constants at the top of
`ScoutServer.java`:** `WHEEL_DIAMETER_MM` and `TRACK_WIDTH_MM`. Measure the
wheels with calipers and the track centre-to-centre of the tyres. Every pose
estimate downstream inherits errors here — a wheel diameter 2% off is a 2%
range error on every wall in your map.

### Gate

- `nxjflash` completes.
- `ProtocolTest` passes with zero failures.
- A test program spins a motor.

---

## Phase 1–2 — Pi setup and the link

### Bluetooth pairing

On the Pi:

```bash
bluetoothctl
```

Then inside `bluetoothctl`: `scan on`, wait for the NXT's address, `pair <ADDR>`
(PIN `1234`), `trust <ADDR>`, `quit`.

Bind the serial device:

```bash
sudo rfcomm bind 0 <NXT_BT_ADDRESS> 1
```

Grant yourself access so you do not need `sudo` every time:

```bash
sudo usermod -aG dialout $USER
```

Log out and back in for the group change to apply.

> BlueZ has deprecated the `rfcomm` tool and the bind does not survive a
> reboot. Add a systemd unit once the robot works, or use
> `BluetoothTransport.wait_for_device()` from a helper script at boot.

### Python dependencies

```bash
cd pi && python3 -m pip install -r requirements.txt
```

### Drive it

Start `ScoutServer` on the brick (it will display "waiting for Bluetooth"),
then on the Pi:

```bash
python3 pi/tools/teleop.py
```

### Gate

- `PING` round-trips.
- **Bumper triggers a stop with the Bluetooth link disconnected.** Safety must
  be local — test it with the Pi unplugged, not just with it running.
- Telemetry streams for 10 minutes with `checksum errors` staying at 0.

---

## Developing without hardware

The whole Pi stack runs against a simulated brick, so you can work on the Mac
with nothing plugged in.

Run the tests:

```bash
python3 pi/tests/test_protocol.py && python3 pi/tests/test_robot.py
```

Drive a simulated robot — in one terminal:

```bash
python3 pi/tools/sim_firmware.py --port 5555
```

and in another:

```bash
python3 pi/tools/teleop.py --sim
```

The simulator mirrors the firmware's wire protocol, ACK/event semantics, and
safety-stop behaviour. It is not a physics simulator — motion is simple
constant-speed kinematics.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Cannot load NXTComm driver` | 64-bit JDK. Install a 32-bit JDK 7 and repoint `JAVA_HOME`. |
| `nxjflash` finds no NXT | Not in firmware update mode. Hold reset >4s until it ticks. |
| Battery empty, buttons dead after flashing | Re-seat one battery. Do not re-flash. |
| Motors behave erratically after flashing | Restart the brick. |
| `/dev/rfcomm0` does not exist | Not bound. `sudo rfcomm bind 0 <ADDR> 1`. |
| Permission denied on `/dev/rfcomm0` | Not in `dialout`, or you have not logged out and back in. |
| Connects, but no telemetry | `ScoutServer` is not running on the brick, or it is in RAW vs PACKET mismatch. |
| `checksum errors` climbing during use | WiFi/Bluetooth coexistence. Lower the camera frame rate first; see the plan's escalation path. |
| Turret grinds or the camera ribbon pulls taut | `TURRET_MAX_ANGLE_DEG` is too large for your build. Lower it. |
