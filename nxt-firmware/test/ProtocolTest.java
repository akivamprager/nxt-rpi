import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import scout.Protocol;

/**
 * Cross-language conformance test for the Scout wire protocol.
 *
 * <p>Replays the golden vectors in {@code vectors.txt}, which are generated
 * from the Python implementation on the Pi side. Passing this proves the
 * firmware's encoder and decoder agree with the Pi byte for byte.
 *
 * <p>This is worth running because a protocol mismatch does not announce
 * itself. It shows up as a robot that silently ignores some commands, or that
 * misreads a pose by a factor of ten - both painful to diagnose over a
 * Bluetooth link with only the NXT's four-line LCD for feedback.
 *
 * <p>Plain Java with no leJOS dependency, so it runs on the same machine you
 * use to build the firmware. From {@code nxt-firmware/}:
 *
 * <pre>
 *   javac -d build/test src/scout/Protocol.java test/ProtocolTest.java
 *   java  -cp build/test ProtocolTest test/vectors.txt
 * </pre>
 *
 * <p>Regenerate the vectors after any protocol change with
 * {@code python3 pi/tools/gen_vectors.py}.
 */
public class ProtocolTest {

    private static int passed;
    private static int failed;

    public static void main(String[] args) throws Exception {
        String path = args.length > 0 ? args[0] : "test/vectors.txt";

        BufferedReader reader = new BufferedReader(new FileReader(path));
        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.length() == 0 || line.charAt(0) == '#') {
                continue;
            }
            String[] parts = line.split("\\s+");
            if (parts[0].equals("FRAME")) {
                checkFrame(parts);
            } else if (parts[0].equals("TELEM")) {
                checkTelemetry(parts);
            } else if (parts[0].equals("STREAM")) {
                checkStream(parts);
            }
        }
        reader.close();

        System.out.println();
        System.out.println(passed + " passed, " + failed + " failed");
        if (failed > 0) {
            System.out.println();
            System.out.println("The Java and Python protocol implementations DISAGREE.");
            System.out.println("Do not flash this firmware until they match.");
        }
        System.exit(failed > 0 ? 1 : 0);
    }

    /** FRAME &lt;name&gt; &lt;type&gt; &lt;payload|-&gt; &lt;encoded&gt; */
    private static void checkFrame(String[] parts) {
        String name = parts[1];
        byte type = (byte) Integer.parseInt(parts[2], 16);
        byte[] payload = unhex(parts[3]);
        byte[] expected = unhex(parts[4]);

        // Encode: build the frame the way ScoutServer.sendFrame does.
        byte[] actual = new byte[Protocol.FRAME_OVERHEAD + payload.length];
        actual[0] = (byte) Protocol.SYNC0;
        actual[1] = (byte) Protocol.SYNC1;
        actual[2] = (byte) payload.length;
        actual[3] = type;
        System.arraycopy(payload, 0, actual, 4, payload.length);
        actual[4 + payload.length] =
                (byte) Protocol.checksum(type, payload, payload.length);

        check("encode " + name, expected, actual);

        // Decode: the same bytes must come back out.
        Protocol.Decoder decoder = new Protocol.Decoder();
        decoder.feed(expected, 0, expected.length);
        if (!decoder.poll()) {
            fail("decode " + name, "no frame decoded");
            return;
        }
        if (decoder.getType() != type) {
            fail("decode " + name,
                    "type " + hex(decoder.getType()) + " != " + hex(type));
            return;
        }
        byte[] got = new byte[decoder.getPayloadLength()];
        System.arraycopy(decoder.getPayload(), 0, got, 0, got.length);
        check("decode " + name, payload, got);
    }

    /**
     * TELEM &lt;seq&gt; &lt;x&gt; &lt;y&gt; &lt;hdg&gt; &lt;turret&gt; &lt;range&gt;
     * &lt;colour&gt; &lt;light&gt; &lt;bumpers&gt; &lt;battery&gt; &lt;flags&gt;
     * &lt;payload&gt;
     *
     * <p>Verifies the exact byte layout ScoutServer.sendTelemetry produces,
     * including the big-endian packing and the fixed-point scaling.
     */
    private static void checkTelemetry(String[] parts) {
        int seq = Integer.parseInt(parts[1]);
        int x = (int) Long.parseLong(parts[2]);
        int y = (int) Long.parseLong(parts[3]);
        int heading = Integer.parseInt(parts[4]);
        int turret = Integer.parseInt(parts[5]);
        int range = Integer.parseInt(parts[6]);
        int colour = Integer.parseInt(parts[7]);
        int light = Integer.parseInt(parts[8]);
        int bumpers = Integer.parseInt(parts[9]);
        int battery = Integer.parseInt(parts[10]);
        int flags = Integer.parseInt(parts[11]);
        byte[] expected = unhex(parts[12]);

        byte[] actual = new byte[Protocol.TELEMETRY_SIZE];
        int i = 0;
        i = Protocol.putInt16(actual, i, seq & 0xFFFF);
        i = Protocol.putInt32(actual, i, x);
        i = Protocol.putInt32(actual, i, y);
        i = Protocol.putInt16(actual, i, heading);
        i = Protocol.putInt16(actual, i, turret);
        actual[i++] = (byte) range;
        actual[i++] = (byte) colour;
        actual[i++] = (byte) light;
        actual[i++] = (byte) bumpers;
        i = Protocol.putInt16(actual, i, battery);
        actual[i++] = (byte) flags;

        check("telemetry seq=" + seq, expected, actual);

        // And the reverse: the unpacking helpers must recover the originals.
        if (Protocol.getInt32(expected, 2) != x) {
            fail("telemetry seq=" + seq,
                    "x round-trip: " + Protocol.getInt32(expected, 2) + " != " + x);
        }
        if (Protocol.getInt32(expected, 6) != y) {
            fail("telemetry seq=" + seq,
                    "y round-trip: " + Protocol.getInt32(expected, 6) + " != " + y);
        }
        if (Protocol.getInt16(expected, 10) != heading) {
            fail("telemetry seq=" + seq,
                    "heading round-trip: " + Protocol.getInt16(expected, 10)
                            + " != " + heading);
        }
    }

    /**
     * STREAM &lt;name&gt; &lt;input&gt; &lt;type:payload,...&gt;
     *
     * <p>The important cases: resync after garbage, surviving a bad checksum,
     * and recovering frames buried behind a corrupted length byte.
     */
    private static void checkStream(String[] parts) {
        String name = parts[1];
        byte[] input = unhex(parts[2]);
        String spec = parts[3];

        List<String> expected = new ArrayList<String>();
        if (!spec.equals("-")) {
            String[] items = spec.split(",");
            for (int i = 0; i < items.length; i++) {
                expected.add(items[i].toUpperCase());
            }
        }

        Protocol.Decoder decoder = new Protocol.Decoder();
        decoder.feed(input, 0, input.length);

        List<String> actual = new ArrayList<String>();
        while (decoder.poll()) {
            byte[] payload = new byte[decoder.getPayloadLength()];
            System.arraycopy(decoder.getPayload(), 0, payload, 0, payload.length);
            actual.add(hex(decoder.getType()) + ":"
                    + (payload.length == 0 ? "-" : hexString(payload)));
        }

        if (!expected.equals(actual)) {
            fail("stream " + name, "expected " + expected + " but got " + actual);
        } else {
            pass("stream " + name);
        }

        // Byte-at-a-time delivery must give the identical result. RFCOMM will
        // fragment frames in practice, so this is not a theoretical concern.
        Protocol.Decoder dribble = new Protocol.Decoder();
        List<String> dribbled = new ArrayList<String>();
        byte[] one = new byte[1];
        for (int i = 0; i < input.length; i++) {
            one[0] = input[i];
            dribble.feed(one, 0, 1);
            while (dribble.poll()) {
                byte[] payload = new byte[dribble.getPayloadLength()];
                System.arraycopy(dribble.getPayload(), 0, payload, 0, payload.length);
                dribbled.add(hex(dribble.getType()) + ":"
                        + (payload.length == 0 ? "-" : hexString(payload)));
            }
        }
        if (!expected.equals(dribbled)) {
            fail("stream " + name + " (byte-at-a-time)",
                    "expected " + expected + " but got " + dribbled);
        } else {
            pass("stream " + name + " (byte-at-a-time)");
        }
    }

    // ---- helpers ----

    private static void check(String name, byte[] expected, byte[] actual) {
        if (expected.length != actual.length) {
            fail(name, "length " + actual.length + " != " + expected.length
                    + "\n    expected " + hexString(expected)
                    + "\n    actual   " + hexString(actual));
            return;
        }
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] != actual[i]) {
                fail(name, "byte " + i + " is " + hex(actual[i])
                        + " but should be " + hex(expected[i])
                        + "\n    expected " + hexString(expected)
                        + "\n    actual   " + hexString(actual));
                return;
            }
        }
        pass(name);
    }

    private static void pass(String name) {
        passed++;
        System.out.println("ok   " + name);
    }

    private static void fail(String name, String detail) {
        failed++;
        System.out.println("FAIL " + name + ": " + detail);
    }

    private static byte[] unhex(String text) {
        if (text.equals("-")) {
            return new byte[0];
        }
        byte[] out = new byte[text.length() / 2];
        for (int i = 0; i < out.length; i++) {
            out[i] = (byte) Integer.parseInt(text.substring(i * 2, i * 2 + 2), 16);
        }
        return out;
    }

    private static String hexString(byte[] data) {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < data.length; i++) {
            sb.append(hex(data[i]));
        }
        return sb.toString();
    }

    private static String hex(byte value) {
        String s = Integer.toHexString(value & 0xFF).toUpperCase();
        return s.length() == 1 ? "0" + s : s;
    }
}
