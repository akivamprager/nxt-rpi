package scout;

import lejos.nxt.NXTRegulatedMotor;

/**
 * The scanning head: one motor carrying both the ultrasonic sensor and the Pi
 * camera.
 *
 * <p><b>The travel limit is a hardware protection, not a preference.</b> The
 * camera's CSI ribbon runs from the rotating head down to the Pi on the
 * chassis. Continuous rotation will wind it up and tear it. Every public entry
 * point clamps to +/-{@code maxAngleDeg}, and there is deliberately no method
 * to bypass the clamp.
 *
 * <p>Angles are turret degrees (what the sensors actually point at), not motor
 * degrees. {@code gearRatio} converts between them: motor degrees = turret
 * degrees * gearRatio. Zero is straight ahead, positive is counter-clockwise,
 * matching the robot's odometry heading convention.
 */
public class Turret {

    private final NXTRegulatedMotor motor;
    private final float gearRatio;
    private final int maxAngleDeg;

    /**
     * @param motor       the turret motor (Motor.C on the reference build)
     * @param gearRatio   motor degrees per turret degree; 1.0 for a direct drive
     * @param maxAngleDeg travel limit either side of centre; keep at or below
     *                    the point where the camera ribbon goes taut
     */
    public Turret(NXTRegulatedMotor motor, float gearRatio, int maxAngleDeg) {
        this.motor = motor;
        this.gearRatio = gearRatio;
        this.maxAngleDeg = maxAngleDeg;
    }

    /**
     * Declares the current physical position to be zero.
     *
     * <p>Call this with the head centred by hand at startup. There is no index
     * switch on the turret, so this is the only thing establishing where centre
     * is - if it is wrong, every bearing in the map is wrong by the same offset.
     */
    public void calibrateHere() {
        motor.resetTachoCount();
    }

    /** Sets the sweep speed in turret degrees per second. */
    public void setSpeed(int degPerSec) {
        motor.setSpeed((int) (degPerSec * gearRatio));
    }

    /** Current head bearing in turret degrees. */
    public float getAngle() {
        return motor.getTachoCount() / gearRatio;
    }

    public boolean isMoving() {
        return motor.isMoving();
    }

    /** The travel limit, so callers can plan a sweep that stays inside it. */
    public int getMaxAngle() {
        return maxAngleDeg;
    }

    /** Clamps a requested bearing into the safe range. */
    public int clamp(int degrees) {
        if (degrees > maxAngleDeg) {
            return maxAngleDeg;
        }
        if (degrees < -maxAngleDeg) {
            return -maxAngleDeg;
        }
        return degrees;
    }

    /**
     * Moves the head to an absolute bearing, clamped to the travel limit.
     *
     * @param degrees         requested turret bearing
     * @param immediateReturn true to start the move and return at once
     * @return the bearing actually commanded after clamping
     */
    public int rotateTo(int degrees, boolean immediateReturn) {
        int target = clamp(degrees);
        motor.rotateTo((int) (target * gearRatio), immediateReturn);
        return target;
    }

    /** Returns the head to centre and blocks until it arrives. */
    public void center() {
        rotateTo(0, false);
    }

    /** Cuts power to the turret motor. */
    public void stop() {
        motor.stop(true);
    }

    /** Releases the motor so the head can be turned by hand. */
    public void relax() {
        motor.flt(true);
    }
}
