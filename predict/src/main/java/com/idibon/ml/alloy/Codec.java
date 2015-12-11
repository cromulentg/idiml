package com.idibon.ml.alloy;

import java.nio.charset.CharsetDecoder;
import java.nio.charset.Charset;
import java.nio.CharBuffer;
import java.nio.ByteBuffer;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.IOException;
import java.io.EOFException;

/**
 * Helper methods for reading and writing compact types to streams
 *
 * Often, we would like to be able to store arbitrary values within Alloy
 * resources (e.g., lengths of strings); however, in practice these values
 * are almost always small enough to fit within 1 or 2 bytes.
 *
 * Therefore, this class provides a number of helper methods for reading
 * and writing values using variable-length encodings to conserve space
 * in the common case.
 */
public final class Codec {

    /**
     * Methods to read and write Strings.
     *
     * Strings are encoded as a sequence of UTF-8-encoded bytes preceded
     * by a variable-length integer (VLuint) specifying the number of bytes
     * in the encoding.
     */
    public static final class String {

        /**
         * Reads an encoded {@link java.lang.String} from an input stream.
         *
         * @param input  stream where the encoded string is stored
         */
        public static java.lang.String read(InputStream input)
                throws IOException {
            int length = Codec.VLuint.read(input);
            byte[] utf8Bytes = new byte[length];

            for (int offset = 0; offset < length; ) {
                int read = input.read(utf8Bytes, offset, length - offset);
                if (read == -1) throw new EOFException("Truncated string");
                offset += read;
            }

            return UTF8_Decoder.get()
                .decode(ByteBuffer.wrap(utf8Bytes))
                .toString();
        }

        /**
         * Encodes and write a {@link java.lang.String} to an output stream
         */
        public static void write(OutputStream output, CharSequence str)
                throws IOException {

            ByteBuffer utf8Buffer = UTF8.encode(CharBuffer.wrap(str));
            byte[] utf8Bytes = new byte[utf8Buffer.remaining()];
            utf8Buffer.get(utf8Bytes);
            Codec.VLuint.write(output, utf8Bytes.length);
            output.write(utf8Bytes);
        }
    }

    /**
     * Methods to read and write variable-length unsigned integers (VLuint).
     *
     * VLint uses a 7/1-bit format encoding, where the 7 LSBs of each byte
     * are the payload, and the MSB indicates a continuation. Values are
     * stored little-endian. This allows integer values [0..127] to be stored
     * in 1 byte, and values [0..16384] to be stored in 2.
     */
    public static final class VLuint {

        // at most 5 bytes are needed to encode a 31-bit value
        private static final int MAX_BYTES = 5;

        /**
         * Reads an unsigned integer from the input stream
         */
        public static int read(InputStream input) throws IOException {
            int val = 0;
            for (int i = 1, shift = 0; i <= MAX_BYTES; i++, shift += 7) {
                int x = input.read();
                if (x == -1) throw new EOFException("Truncated int");
                boolean continuation = (x & 0x80) != 0;
                val |= ((x & 0x7f) << shift);

                // quit if the continuation bit is clear
                if (!continuation) break;
                // continuation set on the last byte = b0rked encoding. bail.
                if (i == MAX_BYTES) throw new IOException("Invalid encoding");
            }
            return val;
        }

        /**
         * Encodes and writes an integer to the output stream
         */
        public static void write(OutputStream output, int value)
                throws IOException {

            if (value < 0) throw new IllegalArgumentException("signed");

            do {
                int b = value & 0x7f;
                value >>= 7;
                // add the continuation bit if needed
                if (value > 0) b |= 0x80;
                output.write(b);
            } while (value > 0);
        }
    }

    // singleton reference to the UTF-8 character set
    private static final Charset UTF8 = Charset.forName("UTF-8");

    /* cached thread-local UTF8 decoders, since CharsetDecoder is
     * not re-entrant, and Charset#decode will not raise an exception
     * on invalid bytestreams (it replaces the invalid bytes). */
    private static final ThreadLocal<CharsetDecoder> UTF8_Decoder =
        new ThreadLocal<CharsetDecoder>() {
            @Override protected CharsetDecoder initialValue() {
                return UTF8.newDecoder();
            }
        };
}
