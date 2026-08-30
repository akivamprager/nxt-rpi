package scout;

import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;

import lejos.nxt.Battery;
import lejos.nxt.Button;
import lejos.nxt.ColorSensor;
import lejos.nxt.LCD;
import lejos.nxt.Motor;
import lejos.nxt.SensorPort;
import lejos.nxt.Sound;
import lejos.nxt.TouchSensor;
import lejos.nxt.UltrasonicSensor;
import lejos.nxt.comm.BTConnection;
import lejos.nxt.comm.Bluetooth;
import lejos.nxt.comm.NXTConnection;
import lejos.robotics.localization.OdometryPoseProvider;
import lejos.robotics.navigation.DifferentialPilot;
import lejos.robotics.navigation.Pose;

/**
 * The robot's body: a dumb, reliable command server.
 *
 * <p>This program deliberately contains no autonomy. It executes motion
 * primitives, streams sensor telemetry, and enforces its own safety stop. All
 * the thinking - vision, localisation, mapping, planning - happens on the Pi.
 * Keeping this layer dumb is what lets it stay stable, which matters because
 * re-uploading it means going back to the Windows laptop.
 *
 * <p><b>Safety is local and authoritative.</b> The bumper and ultrasonic checks
 * run in this loop and stop the motors without consulting the Pi. A safety
 * decision must never depend on a Bluetooth round trip, because the link can
 * stall for hundreds of milliseconds - or drop entirely - and the robot would
 * keep driving into the wall while it waited.
 *
 * <p>Units on the wire are millimetres and deci-degrees (see Protocol). The
 * pilot is constructed in millimetres so its speeds are mm/s and the odometry
 * pose is already in the units the Pi expects.
 */
public class ScoutServer {

    // ---- Robot geometry. MEASURE THESE ON YOUR BUILD. ----
    // Every pose estimate downstream inherits errors here, and a wheel diameter
    // that is 2% off is a 2% range error on every mapped wall. Measure the
    // wheels with calipers and the track width centre-to-centre of the tyres.
    private static final float WHEEL_DIAMETER_MM = 56.0f;
    private static final float TRACK_WIDTH_MM = 115.0f;

    /** Motor degrees per turret degree. 1.0 for a direct-drive head. */
    private static final float TURRET_GEAR_RATIO = 1.0f;

    /** Travel limit either side of centre. Protects the camera ribbon. */
    private static final int TURRET_MAX_ANGLE_DEG = 120;

    // ---- Loop timing. ----
    private static final int LOOP_MS = 20;          // 50 Hz comms + safety
    private static final int SENSOR_PERIOD_MS = 50; // 20 Hz I2C sampling
    private static final int DEFAULT_TELEMETRY_MS = 100;

    // ---- Defaults. ----
    private static final int DEFAULT_SAFETY_RANGE_CM = 20;
    private static final int LOW_BATTERY_MV = 6200;

    private DifferentialPilot pilot;
    private OdometryPoseProvider odometry;
    private Turret turret;
    private UltrasonicSensor sonar;
    private ColorSensor colorSensor;
    private TouchSensor bumperLeft;
    private TouchSensor bumperRight;

    private BTConnection connection;
    private InputStream in;
    private DataOutputStream out;

    private final Protocol.Decoder decoder = new Protocol.Decoder();
    private final byte[] readBuffer = new byte[128];
    private final byte[] frameBuffer = new byte[Protocol.FRAME_OVERHEAD + Protocol.MAX_PAYLOAD];
    private final byte[] telemetryPayload = new byte[Protocol.TELEMETRY_SIZE];

    // ---- Cached sensor state, refreshed at SENSOR_PERIOD_MS. ----
    private int rangeCm = Protocol.US_NO_ECHO;
    private int colorId;
    private int lightLevel;
    private int bumperBits;

    // ---- Mutable configuration, set by the Pi. ----
    private boolean safetyEnabled = true;
    private int safetyRangeCm = DEFAULT_SAFETY_RANGE_CM;
    private int telemetryPeriodMs = DEFAULT_TELEMETRY_MS;

    private boolean safetyTripped;
    private boolean movingForward;
    private boolean awaitingMoveDone;
    private boolean lowBatteryReported;
    private int telemetrySeq;
    private boolean running = true;

    public static void main(String[] args) throws Exception {
        new ScoutServer().run();
    }

    private void run() throws Exception {
        initHardware();

        while (true) {
            LCD.clear();
            LCD.drawString("Scout: waiting", 0, 0);
            LCD.drawString("for Bluetooth", 0, 1);
            LCD.drawString("ESC to quit", 0, 3);

            // RAW mode: leJOS does no framing, we do our own (see Protocol).
            connection = Bluetooth.waitForConnection(0, NXTConnection.RAW);
            if (connection == null) {
                continue;
            }

            in = connection.openInputStream();
            out = connection.openDataOutputStream();

            LCD.clear();
            LCD.drawString("Scout: connected", 0, 0);
            Sound.beepSequenceUp();

            serve();

            closeQuietly();
            haltMotors();
            if (!running) {
                break;
            }
            Sound.beepSequence();
        }

        haltMotors();
        LCD.clear();
        LCD.drawString("Scout: stopped", 0, 0);
    }

    private void initHardware() {
        pilot = new DifferentialPilot(
                WHEEL_DIAMETER_MM, TRACK_WIDTH_MM, Motor.A, Motor.B);
        pilot.setTravelSpeed(150);
        pilot.setRotateSpeed(60);

        odometry = new OdometryPoseProvider(pilot);

        turret = new Turret(Motor.C, TURRET_GEAR_RATIO, TURRET_MAX_ANGLE_DEG);
        turret.setSpeed(90);
        // No index switch exists, so "wherever the head is now" defines centre.
        // Centre it by hand before starting the program.
        turret.calibrateHere();

        sonar = new UltrasonicSensor(SensorPort.S1);
        colorSensor = new ColorSensor(SensorPort.S2);
        bumperLeft = new TouchSensor(SensorPort.S3);
        bumperRight = new TouchSensor(SensorPort.S4);
    }

    /** Main loop for one Bluetooth session. Returns when the link drops. */
    private void serve() {
        long nextSensor = 0;
        long nextTelemetry = 0;

        while (true) {
            long now = System.currentTimeMillis();

            if (Button.ESCAPE.isDown()) {
                running = false;
                return;
            }

            if (now >= nextSensor) {
                nextSensor = now + SENSOR_PERIOD_MS;
                sampleSensors();
                enforceSafety();
            }

            if (!pumpInput()) {
                return; // link dropped
            }

            checkMoveCompletion();

            if (telemetryPeriodMs > 0 && now >= nextTelemetry) {
                nextTelemetry = now + telemetryPeriodMs;
                if (!sendTelemetry()) {
                    return;
                }
            }

            long slack = LOOP_MS - (System.currentTimeMillis() - now);
            if (slack > 0) {
                try {
                    Thread.sleep(slack);
                } catch (InterruptedException ignored) {
                    // Nothing else runs on this thread; resuming is correct.
                }
            }
        }
    }

    private void sampleSensors() {
        rangeCm = sonar.getDistance();
        colorId = colorSensor.getColorID();
        lightLevel = colorSensor.getLightValue();

        bumperBits = 0;
        if (bumperLeft.isPressed()) {
            bumperBits |= 0x01;
        }
        if (bumperRight.isPressed()) {
            bumperBits |= 0x02;
        }

        if (!lowBatteryReported && Battery.getVoltageMilliVolt() < LOW_BATTERY_MV) {
            lowBatteryReported = true;
            sendEvent(Protocol.EV_LOW_BATTERY);
        }
    }

    /**
     * Stops the robot on a bumper hit or an obstacle inside the safety range.
     *
     * <p>Runs locally, every sensor tick, with no dependency on the Pi.
     *
     * <p>Only forward motion is inhibited. Both sensors face forward, so
     * blocking reverse would strand the robot against whatever it just hit with
     * no way to back off.
     */
    private void enforceSafety() {
        if (bumperBits != 0) {
            if (movingForward) {
                haltMotors();
                sendEvent(Protocol.EV_BUMPER);
            }
            safetyTripped = true;
            return;
        }

        // US_NO_ECHO means "learned nothing", not "clear" - do not treat it as
        // an obstacle or the robot would freeze whenever the beam scatters off
        // an angled wall.
        boolean blocked = safetyEnabled
                && rangeCm != Protocol.US_NO_ECHO
                && rangeCm <= safetyRangeCm;

        if (blocked && movingForward) {
            haltMotors();
            sendEvent(Protocol.EV_SAFETY_STOP);
        }
        safetyTripped = blocked;
    }

    /** Reports completion of a blocking-style move started with immediateReturn. */
    private void checkMoveCompletion() {
        if (awaitingMoveDone && !pilot.isMoving()) {
            awaitingMoveDone = false;
            movingForward = false;
            sendEvent(Protocol.EV_MOVE_DONE);
        }
    }

    /**
     * Drains available input and dispatches complete frames.
     *
     * @return false if the connection has closed
     */
    private boolean pumpInput() {
        try {
            int available = in.available();
            while (available > 0) {
                int n = in.read(readBuffer, 0,
                        available < readBuffer.length ? available : readBuffer.length);
                if (n < 0) {
                    return false;
                }
                decoder.feed(readBuffer, 0, n);
                available -= n;
            }
        } catch (IOException e) {
            return false;
        }

        while (decoder.poll()) {
            dispatch(decoder.getType(), decoder.getPayload(), decoder.getPayloadLength());
        }
        return true;
    }

    private void dispatch(byte type, byte[] payload, int length) {
        switch (type) {
            case Protocol.CMD_PING:
                sendFrame(Protocol.RSP_PONG, null, 0);
                break;

            case Protocol.CMD_STOP:
                haltMotors();
                turret.stop();
                awaitingMoveDone = false;
                ack(type, Protocol.ACK_OK);
                break;

            case Protocol.CMD_DRIVE:
                if (length != 4) {
                    ack(type, Protocol.ACK_BAD_LENGTH);
                    break;
                }
                doDrive(Protocol.getInt16(payload, 0),
                        Protocol.getInt16(payload, 2) / 10.0f);
                break;

            case Protocol.CMD_TRAVEL:
                if (length != 4) {
                    ack(type, Protocol.ACK_BAD_LENGTH);
                    break;
                }
                doTravel(Protocol.getInt32(payload, 0));
                break;

            case Protocol.CMD_TURN_TO:
                if (length != 2) {
                    ack(type, Protocol.ACK_BAD_LENGTH);
                    break;
                }
                doTurnTo(Protocol.getInt16(payload, 0) / 10.0f);
                break;

            case Protocol.CMD_TURRET_TO:
                if (length != 2) {
                    ack(type, Protocol.ACK_BAD_LENGTH);
                    break;
                }
                turret.rotateTo(Math.round(Protocol.getInt16(payload, 0) / 10.0f), true);
                ack(type, Protocol.ACK_OK);
                break;

            case Protocol.CMD_SET_POSE:
                if (length != 10) {
                    ack(type, Protocol.ACK_BAD_LENGTH);
                    break;
                }
                // How the Pi corrects odometry drift after an ArUco fix.
                odometry.setPose(new Pose(
                        Protocol.getInt32(payload, 0),
                        Protocol.getInt32(payload, 4),
                        Protocol.getInt16(payload, 8) / 10.0f));
                ack(type, Protocol.ACK_OK);
                break;

            case Protocol.CMD_SET_SAFETY:
                if (length != 2) {
                    ack(type, Protocol.ACK_BAD_LENGTH);
                    break;
                }
                safetyEnabled = payload[0] != 0;
                safetyRangeCm = payload[1] & 0xFF;
                ack(type, Protocol.ACK_OK);
                break;

            case Protocol.CMD_SET_TELEMETRY:
                if (length != 2) {
                    ack(type, Protocol.ACK_BAD_LENGTH);
                    break;
                }
                telemetryPeriodMs = ((payload[0] & 0xFF) << 8) | (payload[1] & 0xFF);
                ack(type, Protocol.ACK_OK);
                break;

            default:
                ack(type, Protocol.ACK_UNKNOWN_CMD);
                break;
        }
    }

    // ---- Motion primitives ----

    /**
     * Continuous velocity control, used for teleop.
     *
     * @param vMmS     forward speed, mm/s
     * @param omegaDps rotation rate, degrees/s, positive counter-clockwise
     */
    private void doDrive(int vMmS, float omegaDps) {
        if (vMmS > 0 && !canMoveForward()) {
            ack(Protocol.CMD_DRIVE, Protocol.ACK_REFUSED);
            return;
        }

        awaitingMoveDone = false;
        movingForward = vMmS > 0;

        if (vMmS == 0 && omegaDps == 0) {
            haltMotors();
        } else if (omegaDps == 0) {
            pilot.setTravelSpeed(Math.abs(vMmS));
            if (vMmS > 0) {
                pilot.forward();
            } else {
                pilot.backward();
            }
        } else if (vMmS == 0) {
            // Spin in place. A large angle with immediateReturn approximates
            // continuous rotation; STOP or the next command ends it.
            pilot.setRotateSpeed(Math.abs(omegaDps));
            pilot.rotate(omegaDps > 0 ? 3600 : -3600, true);
        } else {
            // Arc of radius v / omega, converting omega to radians/s.
            float radius = (float) (vMmS / (omegaDps * Math.PI / 180.0));
            pilot.setTravelSpeed(Math.abs(vMmS));
            pilot.arcForward(radius);
        }
        ack(Protocol.CMD_DRIVE, Protocol.ACK_OK);
    }

    private void doTravel(int distanceMm) {
        if (distanceMm > 0 && !canMoveForward()) {
            ack(Protocol.CMD_TRAVEL, Protocol.ACK_REFUSED);
            return;
        }
        movingForward = distanceMm > 0;
        awaitingMoveDone = true;
        pilot.travel(distanceMm, true);
        ack(Protocol.CMD_TRAVEL, Protocol.ACK_OK);
    }

    /** Turns to an absolute heading in the odometry frame. */
    private void doTurnTo(float headingDeg) {
        float delta = Protocol.normaliseDegrees(
                headingDeg - odometry.getPose().getHeading());
        movingForward = false; // rotation in place is never inhibited
        awaitingMoveDone = true;
        pilot.rotate(delta, true);
        ack(Protocol.CMD_TURN_TO, Protocol.ACK_OK);
    }

    /**
     * Whether forward motion is currently permitted.
     *
     * <p>Refusing the command outright (rather than accepting and immediately
     * stopping) gives the Pi an unambiguous ACK_REFUSED to plan against.
     */
    private boolean canMoveForward() {
        if (bumperBits != 0) {
            return false;
        }
        return !(safetyEnabled
                && rangeCm != Protocol.US_NO_ECHO
                && rangeCm <= safetyRangeCm);
    }

    private void haltMotors() {
        if (pilot != null) {
            pilot.stop();
        }
        movingForward = false;
    }

    // ---- Outbound frames ----

    private boolean sendTelemetry() {
        Pose pose = odometry.getPose();

        int flags = 0;
        if (pilot.isMoving()) {
            flags |= Protocol.FLAG_MOVING;
        }
        if (turret.isMoving()) {
            flags |= Protocol.FLAG_TURRET_MOVING;
        }
        if (safetyTripped) {
            flags |= Protocol.FLAG_SAFETY_TRIPPED;
        }
        if (safetyEnabled) {
            flags |= Protocol.FLAG_SAFETY_ENABLED;
        }

        int battery = Battery.getVoltageMilliVolt();
        int light = lightLevel;
        if (light < 0) {
            light = 0;
        } else if (light > 255) {
            light = 255;
        }

        int i = 0;
        i = Protocol.putInt16(telemetryPayload, i, telemetrySeq++ & 0xFFFF);
        i = Protocol.putInt32(telemetryPayload, i, Math.round(pose.getX()));
        i = Protocol.putInt32(telemetryPayload, i, Math.round(pose.getY()));
        i = Protocol.putInt16(telemetryPayload, i,
                Math.round(Protocol.normaliseDegrees(pose.getHeading()) * 10.0f));
        i = Protocol.putInt16(telemetryPayload, i, Math.round(turret.getAngle() * 10.0f));
        telemetryPayload[i++] = (byte) rangeCm;
        telemetryPayload[i++] = (byte) colorId;
        telemetryPayload[i++] = (byte) light;
        telemetryPayload[i++] = (byte) bumperBits;
        i = Protocol.putInt16(telemetryPayload, i, battery);
        telemetryPayload[i++] = (byte) flags;

        return sendFrame(Protocol.RSP_TELEMETRY, telemetryPayload, Protocol.TELEMETRY_SIZE);
    }

    private void sendEvent(byte eventCode) {
        byte[] payload = new byte[1];
        payload[0] = eventCode;
        sendFrame(Protocol.RSP_EVENT, payload, 1);
    }

    private void ack(byte command, byte status) {
        byte[] payload = new byte[2];
        payload[0] = command;
        payload[1] = status;
        sendFrame(Protocol.RSP_ACK, payload, 2);
    }

    /** @return false if the write failed, meaning the link is gone */
    private boolean sendFrame(byte type, byte[] payload, int length) {
        if (out == null) {
            return false;
        }
        frameBuffer[0] = (byte) Protocol.SYNC0;
        frameBuffer[1] = (byte) Protocol.SYNC1;
        frameBuffer[2] = (byte) length;
        frameBuffer[3] = type;
        if (length > 0) {
            System.arraycopy(payload, 0, frameBuffer, 4, length);
        }
        frameBuffer[4 + length] = (byte) Protocol.checksum(type, payload, length);

        try {
            out.write(frameBuffer, 0, Protocol.FRAME_OVERHEAD + length);
            // Flush every frame. Buffered telemetry is stale telemetry, and the
            // Pi's control decisions are only as fresh as what actually shipped.
            out.flush();
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    private void closeQuietly() {
        try {
            if (out != null) {
                out.close();
            }
            if (in != null) {
                in.close();
            }
            if (connection != null) {
                connection.close();
            }
        } catch (IOException ignored) {
            // Already tearing down; nothing useful to do.
        }
        out = null;
        in = null;
        connection = null;
    }
}
