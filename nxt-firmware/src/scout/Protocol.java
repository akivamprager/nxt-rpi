package scout;

/**
 * Scout wire protocol - NXT side.
 *
 * <p>This file MUST stay byte-for-byte in sync with {@code pi/scout/protocol.py}.
 * If you change an opcode or a payload layout here, change it there in the same
 * commit. The Python test suite is the executable specification; this is the
 * mirror.
 *
 * <p>Framing: {@code A5 5A LEN TYPE PAYLOAD[LEN] XOR}, all multi-byte integers
 * big-endian (which is what Java does natively, so the firmware needs no byte
 * swapping).
 *
 * <p>Written for leJOS NXJ 0.9.1, which is a Java 1.5-era environment with a
 * cut-down class library. Deliberately avoids generics, java.nio, collections
 * and String.format - plain arrays and manual byte packing only. Buffers are
 * kept small because the NXT has 64 KB of RAM in total.
 */
public final class Protocol {

    private Protocol() {
    }

    public static final int SYNC0 = 0xA5;
    public static final int SYNC1 = 0x5A;

    /** Sync(2) + LEN(1) + TYPE(1) + XOR(1). */
    public static final int FRAME_OVERHEAD = 5;

    public static final int MAX_PAYLOAD = 255;

    // ---- Pi -> NXT (commands). High bit clear. ----
    public static final byte CMD_PING = 0x01;
    public static final byte CMD_DRIVE = 0x02;
    public static final byte CMD_TRAVEL = 0x03;
    public static final byte CMD_TURN_TO = 0x04;
    public static final byte CMD_STOP = 0x05;
    public static final byte CMD_TURRET_TO = 0x06;
    public static final byte CMD_SET_POSE = 0x07;
    public static final byte CMD_SET_SAFETY = 0x08;
    public static final byte CMD_SET_TELEMETRY = 0x09;

    // ---- NXT -> Pi (responses). High bit set. ----
    public static final byte RSP_PONG = (byte) 0x81;
    public static final byte RSP_TELEMETRY = (byte) 0x82;
    public static final byte RSP_EVENT = (byte) 0x83;
    public static final byte RSP_ACK = (byte) 0x84;
    public static final byte RSP_LOG = (byte) 0x85;

    // ---- Event codes carried by RSP_EVENT. ----
    public static final byte EV_BUMPER = 0x01;
    public static final byte EV_SAFETY_STOP = 0x02;
    public static final byte EV_MOVE_DONE = 0x03;
    public static final byte EV_TURRET_DONE = 0x04;
    public static final byte EV_STALL = 0x05;
    public static final byte EV_LOW_BATTERY = 0x06;

    // ---- ACK status codes. ----
    public static final byte ACK_OK = 0x00;
    public static final byte ACK_BAD_LENGTH = 0x01;
    public static final byte ACK_UNKNOWN_CMD = 0x02;
    public static final byte ACK_REFUSED = 0x03;

    /**
     * leJOS returns 255 from getDistance() when no echo came back. That means
     * "no information", NOT "the way is clear" - the Pi's mapping code treats
     * it as unknown rather than as free space.
     */
    public static final int US_NO_ECHO = 255;

    /** Telemetry payload size: seq(2) x(4) y(4) hdg(2) turret(2) rng(1)
     *  colour(1) light(1) bumpers(1) battery(2) flags(1). */
    public static final int TELEMETRY_SIZE = 21;

    // ---- Telemetry flag bits. ----
    public static final int FLAG_MOVING = 0x01;
    public static final int FLAG_TURRET_MOVING = 0x02;
    public static final int FLAG_SAFETY_TRIPPED = 0x04;
    public static final int FLAG_SAFETY_ENABLED = 0x08;

    /** XOR of the type byte and every payload byte. */
    public static int checksum(byte type, byte[] payload, int length) {
        int value = type & 0xFF;
        for (int i = 0; i < length; i++) {
            value ^= (payload[i] & 0xFF);
        }
        return value & 0xFF;
    }

    // ---- Big-endian packing helpers. ----

    public static int putInt16(byte[] buf, int offset, int value) {
        buf[offset] = (byte) ((value >> 8) & 0xFF);
        buf[offset + 1] = (byte) (value & 0xFF);
        return offset + 2;
    }

    public static int putInt32(byte[] buf, int offset, int value) {
        buf[offset] = (byte) ((value >> 24) & 0xFF);
        buf[offset + 1] = (byte) ((value >> 16) & 0xFF);
        buf[offset + 2] = (byte) ((value >> 8) & 0xFF);
        buf[offset + 3] = (byte) (value & 0xFF);
        return offset + 4;
    }

    /** Reads a signed 16-bit big-endian integer. */
    public static int getInt16(byte[] buf, int offset) {
        return (short) (((buf[offset] & 0xFF) << 8) | (buf[offset + 1] & 0xFF));
    }

    /** Reads a signed 32-bit big-endian integer. */
    public static int getInt32(byte[] buf, int offset) {
        return ((buf[offset] & 0xFF) << 24)
                | ((buf[offset + 1] & 0xFF) << 16)
                | ((buf[offset + 2] & 0xFF) << 8)
                | (buf[offset + 3] & 0xFF);
    }

    /** Normalises degrees into [-180, 180), matching what the Pi expects. */
    public static float normaliseDegrees(float degrees) {
        while (degrees >= 180.0f) {
            degrees -= 360.0f;
        }
        while (degrees < -180.0f) {
            degrees += 360.0f;
        }
        return degrees;
    }

    /**
     * Incremental frame decoder for a lossy byte stream.
     *
     * <p>Mirrors {@code FrameDecoder} in protocol.py, including the recovery
     * behaviour: a bad checksum drops only the sync word, not the whole claimed
     * frame, so good frames swallowed by a corrupted length byte can still be
     * recovered.
     *
     * <p>Single-frame-at-a-time by design: the caller polls {@link #poll()} in
     * its main loop, so no allocation happens per frame.
     */
    public static final class Decoder {

        /** Generous enough for any command we define, small enough for the NXT. */
        private static final int BUFFER_SIZE = 320;

        private final byte[] buffer = new byte[BUFFER_SIZE];
        private int count;

        private byte frameType;
        private final byte[] payload = new byte[MAX_PAYLOAD];
        private int payloadLength;

        public int checksumErrors;
        public int resyncs;
        public int overflows;

        /** Appends received bytes. Silently drops the oldest data on overflow. */
        public void feed(byte[] data, int offset, int length) {
            if (length > BUFFER_SIZE) {
                // Keep only the newest bytes; older ones are unrecoverable anyway.
                offset += (length - BUFFER_SIZE);
                length = BUFFER_SIZE;
            }
            if (count + length > BUFFER_SIZE) {
                int drop = count + length - BUFFER_SIZE;
                System.arraycopy(buffer, drop, buffer, 0, count - drop);
                count -= drop;
                overflows++;
            }
            System.arraycopy(data, offset, buffer, count, length);
            count += length;
        }

        /**
         * Attempts to extract one frame.
         *
         * @return true if a frame is ready in {@link #getType()} /
         *         {@link #getPayload()}. Call repeatedly until it returns false.
         */
        public boolean poll() {
            while (true) {
                int limit = count - 1;
                int start = 0;
                while (start < limit
                        && !((buffer[start] & 0xFF) == SYNC0
                                && (buffer[start + 1] & 0xFF) == SYNC1)) {
                    start++;
                }

                if (start >= limit) {
                    // No sync word present. Keep at most one trailing byte: it
                    // may be the first half of a sync word still in flight.
                    if (count > 1) {
                        buffer[0] = buffer[count - 1];
                        count = 1;
                        resyncs++;
                    }
                    return false;
                }

                if (start > 0) {
                    consume(start);
                    resyncs++;
                }

                if (count < FRAME_OVERHEAD) {
                    return false;
                }

                int length = buffer[2] & 0xFF;
                int total = FRAME_OVERHEAD + length;
                if (count < total) {
                    return false;
                }

                byte type = buffer[3];
                System.arraycopy(buffer, 4, payload, 0, length);

                if ((buffer[4 + length] & 0xFF) != checksum(type, payload, length)) {
                    checksumErrors++;
                    consume(2);
                    continue;
                }

                consume(total);
                frameType = type;
                payloadLength = length;
                return true;
            }
        }

        private void consume(int n) {
            System.arraycopy(buffer, n, buffer, 0, count - n);
            count -= n;
        }

        public byte getType() {
            return frameType;
        }

        public byte[] getPayload() {
            return payload;
        }

        public int getPayloadLength() {
            return payloadLength;
        }
    }
}
