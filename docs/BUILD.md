# Physical build

Start from LEGO's **Quick Start model** — the official 20-step build in the
[8547 User Guide](https://www.lego.com/en-us/service/building-instructions/8547)
(pages 5–15, steps 1–20). Build it exactly as documented first. Do not try to
design a chassis from scratch; it already solves the hard part correctly, and
it turns out to solve it in a way that fits our project almost perfectly.

Everything below is what to change once the stock build is standing.

## Why this build is a near-perfect match

Steps 1–13 build **two mirrored motor+track pods** (left and right — this is
your differential drive, Motor A and B). Steps 14–19 join them with wheels,
tracks, and corner brackets, and mount the NXT brick on top. Step 20 wires all
three motors to the brick.

The third motor sits structurally separate from the two track pods — in
LEGO's fuller MINDSTORMS designs this motor is meant to drive an action
attachment (a ball shooter, a gripper, an arm), not a third track. The stock
Quick Start build stops before attaching anything to it, so its output shaft
is sitting there free and unused. **That's our turret motor**, already in
place, already wired to Motor C on the brick. We're not repurposing a busy
motor — we're using the free one for the job it was structurally built to do.

This is exactly the port layout the firmware already assumes (see the table
below), so there is no mismatch to design around.

## Port wiring — not a free choice

The firmware hardcodes these assignments in
[ScoutServer.java](../nxt-firmware/src/scout/ScoutServer.java). Wire the robot
to match, or edit the constants at the top of that file to match your wiring
— but the two must agree.

| Port | Assigned to |
|---|---|
| Motor A | Left drive pod |
| Motor B | Right drive pod |
| Motor C | Turret (already free in the stock build) |
| Sensor 1 | Ultrasonic sensor — **on the turret** |
| Sensor 2 | Color sensor |
| Sensor 3 | Left bumper (touch sensor) |
| Sensor 4 | Right bumper (touch sensor) |

`DifferentialPilot`'s first motor argument is the left wheel — swap A/B in the
firmware constructor if your build ends up mirrored rather than rewiring the
brick.

## Modifications from the stock Quick Start build

**1. Stop at step 20 and don't build an attachment for the third motor.** The
stock instructions end at the bare vehicle with three motors wired and
nothing attached to the third one's output. Leave it that way — that free
shaft is exactly what the turret mounts onto.

**2. Build a turret platform on the third motor.** A small rotating deck on
its output shaft, direct-driven (1:1, no gearing — this matches
`TURRET_GEAR_RATIO = 1.0` in the firmware; if you gear it down for more
torque, update that constant or every angle the Pi computes will be wrong by
the gear ratio). Mount the ultrasonic sensor and a bracket for the Pi Camera
Module on this deck, side by side, both facing the same direction.

**3. Add the ultrasonic sensor to the turret.** The stock build has no
sensors at all — the User Guide's sensor pages (30–33) are reference material
on what each sensor does and which port it plugs into, not a mounting
diagram. There is nothing to "undo" here, just something to add: connect it
on the turret deck, wired to Sensor 1.

**4. Add front bumpers.** Two touch sensors (Sensor 3, Sensor 4) at the very
front edge, left and right of center, each behind a small lever or bumper bar
wide enough that a collision from either side of the front reliably presses
one. The firmware's safety stop depends on these firing on contact.

**5. Add a Pi mounting bay — low, centered, secure.**

- **Position:** low and between the two drive pods, not up top or
  overhanging. The Pi (with a case) plus a power bank is roughly 250 g, and
  mounting that mass high or off-axis makes the robot tip on turns.
- **Securing it:** LEGO has no native way to clamp a rectangular PCB. Build a
  simple cradle — four vertical pins or a beam frame around the Pi's edges so
  it can't slide, then a zip tie or rubber band across the top through nearby
  Technic holes so it can't lift out. Avoid anything that touches the board's
  underside components.
- **Power bank:** strap it alongside or below the Pi the same way. Use a
  2.4 A output; a ~10000 mAh bank gives roughly 6–8 hours.
- **Keep power domains separate.** The NXT keeps its own 6×AA pack (stock);
  the Pi runs off the power bank. No electrical connection between them —
  this is deliberate, and part of why Bluetooth was chosen over a USB tether.

**6. Route the CSI ribbon cable with real slack.** The ribbon runs from the Pi
(mounted low) up to the camera (mounted on the rotating turret, up top). Buy a
30–50 cm ribbon rather than using the stock short one. Leave a generous
service loop at the turret joint specifically — that's the only point that
actually flexes — and zip-tie the rest of the run along a fixed path.
**Physically confirm the ribbon has slack through the full ±120° turret sweep
before powering anything on.** The firmware clamps turret travel to protect
this cable (`TURRET_MAX_ANGLE_DEG` in `ScoutServer.java`), but that only
helps if the physical build actually has clearance to match — verify by hand,
not by trusting the software limit alone.

**7. Verify turret clearance.** With everything mounted, hand-rotate the
turret through its full ±120° sweep and confirm the camera/sensor assembly
doesn't strike the chassis, the antenna, or its own cable at either extreme.
If it does, lower `TURRET_MAX_ANGLE_DEG` to whatever your build can actually
clear.

## Tracks vs. wheels — one tradeoff worth knowing

The stock build uses tank treads, which skid-steer. That's mechanically
noisier for odometry than a pair of free-rolling wheels — the whole reason
Phase 3 adds ArUco correction is that odometry drifts regardless, so treads
will work, but if your kit has enough wheel/tire pieces to convert the two
drive pods to wheels instead, it'll track straighter and give the vision
system less drift to correct. Not required; worth doing if it's easy.

## Weight and balance

Keep the Pi and power bank low and centered. Watch two failure modes once the
turret is loaded with the camera:

- **Too little weight over the rear of the tracks** → the robot doesn't track
  straight and wanders under its own vibration.
- **Too much weight forward** → reduced traction and a tendency to nose-dive
  under braking.

If the turret's added weight (camera + sensor + bracket) noticeably shifts
the balance, compensate by moving the Pi/battery bay, not by fighting it in
software.

## Before Phase 1 (uploading firmware)

Measure the actual wheel/track drive diameter and the track width (center to
center of the two drive pods) with calipers, and update `WHEEL_DIAMETER_MM`
and `TRACK_WIDTH_MM` at the top of `ScoutServer.java`. Every pose estimate
downstream inherits these numbers; a diameter that's 2% off is a 2% range
error on every wall the robot ever maps. If you converted to wheels per the
note above, measure those instead of the stock track sprockets.

## Note for later (Phase 3, not now)

Once the camera is mounted, measure its height off the floor and its forward
offset from the turret's rotation axis — you'll need both to convert a
detected ArUco marker's camera-frame pose into the robot's world pose in
`config.yaml`. Worth jotting down now while you have calipers in hand.
