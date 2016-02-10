package com.idibon.ml.alloy;

import com.idibon.ml.predict.Label;
import com.idibon.ml.predict.PredictOptions;
import com.idibon.ml.predict.PredictResult;
import org.json4s.JsonAST;

import java.io.IOException;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.List;

/**
 * Interface for interacting with Alloys
 *
 * Provides model, feature pipeline, and task configuration I/O to
 * the prediction and training engines.
 */
public interface Alloy<T extends PredictResult> {

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

    /** Returns the full list of labels known by the Alloy */
    public List<Label> getLabels();

    /**
     * Predict a document. Types here are subject to CHANGE!
     * @param document
     * @param options
     * @return
     */
    public List<T> predict(JsonAST.JObject document, PredictOptions options);

    /**
     * Top level method to save an alloy.
     * @param path
     * @throws IOException
     */
    public void save(Alloy.Writer writer) throws IOException;

    /**
     * Translates a UUID of a label into a human readable string.
     * @param uuid the UUID for a label.
     * @return human readable label object.
     */
    public Label translateUUID(String uuid);
}
