package com.idibon.ml.alloy;

import java.io.IOException;
import java.io.DataInputStream;
import java.io.DataOutputStream;

/**
 * Interface for interacting with Alloys
 *
 * Provides model, feature pipeline, and task configuration I/O to
 * the prediction and training engines.
 */
public interface Alloy {

    /**
     * Operations for reading objects from an Alloy.
     */
    public interface Reader {

        /**
         * Returns an Alloy.Reader that only loads resources within
         * the provided namespace.
         */
        Reader within(String namespace) throws IOException;

        /**
         * Returns an input stream to read the specified resource from
         * the current namespace.
         *
         * Callers are responsible for ensuring that all returned streams
         * are closed.
         */
        DataInputStream resource(String resourceName) throws IOException;
    }

    /**
     * Interface for writing objects into an Alloy.
     */
    public interface Writer {
        /**
         * Returns an Alloy.Writer that only writes resources within
         * the provided namespace.
         */
        Writer within(String namespace) throws IOException;

        /**
         * Creates an output stream to store a resource with the specified
         * name within the current namespace.
         *
         * Callers are responsible for ensuring that all returned streams
         * are closed.
         */
        DataOutputStream resource(String resourceName) throws IOException;
    }
}
