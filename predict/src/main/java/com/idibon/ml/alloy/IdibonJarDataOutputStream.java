package com.idibon.ml.alloy;

import java.io.*;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;

/**
 * Wrapper class to allow "concurrent" writing to the Jar being
 * written to.
 *
 * The premise is that we do a bait and switch. We pretend to give
 * them an object that writes to the JAR but in fact, it's just
 * a ByteArrayOutputStream. This allows multiple threads to write
 * in parallel. Then, as good citizens, they try to close the stream,
 * we then actually take what they've written and shove it under the
 * specified JarEntry in the Jar, accounting for the fact that only
 * one thread can write at any one time by "synchronizing" access to
 * the JarOutputStream.
 * Is that ^ correct Gary?
 *
 * @author "Stefan Krawczyk <stefan@idibon.com>"
 */
public class IdibonJarDataOutputStream extends DataOutputStream {

    private JarOutputStream _jos;
    private ByteArrayOutputStream _baos;
    private JarEntry _je;
    /**
     * Creates a new data output stream to write data to the specified
     * underlying output stream. The counter <code>written</code> is
     * set to zero.
     *
     * @param out the underlying output stream, to be saved for later
     *            use.
     * @param jos the stream to actually write to, to affect the JAR file.
     * @param je the jar entry to create for writing to the jar.
     * @see FilterOutputStream#out
     */
    public IdibonJarDataOutputStream(ByteArrayOutputStream out, JarOutputStream jos, JarEntry je) {
        super(out);
        _jos = jos;
        _baos = out;
        _je = je;
    }

    /**
     * Override the close method to actually write to the JAR file.
     * @throws IOException
     */
    @Override public void close() throws IOException {
        _je.setTime(System.currentTimeMillis());
        // we only every want one of these instances to write to the single output stream that
        // writes to the JAR. I believe that is all I need to do?
        synchronized (_jos) {
            _jos.putNextEntry(_je);
            _jos.write(_baos.toByteArray());
            _jos.closeEntry();
        }
        _baos.close();
    }
}
