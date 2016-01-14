package com.idibon.ml.alloy;

import com.idibon.ml.predict.PredictModel;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.jar.*;

/**
 * Created by stefankrawczyk on 1/14/16.
 */
public class JarAlloy2 extends BaseAlloy {


    public JarAlloy2(Map<String, PredictModel> labelModelMap, Map<String, String> labelToUUID) {
        super(labelModelMap, labelToUUID);
    }

    @Override public boolean save(String path) throws IOException{
        // create manifest
        Manifest manifest = new Manifest();
        fillManifest(manifest);

        // create JarOutputStream
        JarOutputStream jos = new JarOutputStream(new FileOutputStream(new File(path)), manifest);
        // save metadata about this task
        // TODO
        // save models
        // for each model
        JarWriter baseWriter = new JarWriter("", jos);
        // JarWriter writer = baseWriter.namespace("label");
        // model.save(writer)

        jos.close();
        return false;
    }

    private void fillManifest(Manifest manifest) {
        Attributes attr = manifest.getMainAttributes();
        attr.put(Attributes.Name.MANIFEST_VERSION, "0.0.1");
        attr.put(new Attributes.Name("Created-By"), "Idibon Inc.");
        attr.put(new Attributes.Name("JVM-Version"), System.getProperty("java.version"));
        attr.put(Attributes.Name.SPECIFICATION_TITLE, "Idibon IdiJar-Alloy for Prediction");
        attr.put(Attributes.Name.SPECIFICATION_VENDOR, "Idibon Inc.");
        attr.put(Attributes.Name.SPECIFICATION_VERSION, "0.0.1");
        attr.put(Attributes.Name.IMPLEMENTATION_TITLE, "com.idibon.ml");
        attr.put(Attributes.Name.IMPLEMENTATION_VENDOR, "Idibon Inc.");
        attr.put(Attributes.Name.IMPLEMENTATION_VERSION, "0.0.1");
    }

    public static JarAlloy2 load(String path) throws IOException {
        File jarFile = new File(path);
        JarFile jar = new JarFile(jarFile);
        // read top level models & get classes

        // using reflection create that class and call the load method and reify models.
        Map<String, PredictModel> labelModels = new HashMap<String, PredictModel>();
        Map<String, String> labelToUUID = new HashMap<String, String>();
        // instantiate other objects

        jar.close();
        // return fresh instance
        return new JarAlloy2(labelModels, labelToUUID);
    }

    class JarWriter implements Alloy.Writer {

        private final JarOutputStream _jos;
        private final String _currentPath;

        /**
         * Constructor that sets up from what path a Writer will start writing at.
         *
         * @param path
         */
        public JarWriter(String path, JarOutputStream jarOutputStream) {
            _currentPath = path;
            _jos = jarOutputStream;
        }

        /**
         * Returns a writer that will ensure that the "namespace" exists and is ready for use.
         *
         * Essentially it just creates a directory entry and returns a new JarWriter to write
         * from that directory as its base.
         *
         * @param namespace
         * @return
         * @throws IOException
         */
        @Override public Writer within(String namespace) throws IOException {
            String namespacePath = _currentPath + namespace + "/";
            JarEntry je = new JarEntry(namespacePath);
            // people need to take turns writing to the JarOutputStream.
            synchronized (_jos) {
                _jos.putNextEntry(je);
                _jos.closeEntry();
            }
            return new JarWriter(namespacePath, _jos);
        }

        /**
         * Use this to get a new DataOutputStream. Remember to use closeResource()
         * once you're done on this writer object.
         *
         * @param resourceName
         * @return
         * @throws IOException
         */
        @Override public DataOutputStream resource(String resourceName) throws IOException {
            JarEntry je = new JarEntry(_currentPath + resourceName); // "/" is taken care of.
            return new IdibonJarDataOutputStream(new ByteArrayOutputStream(), _jos, je);
        }
    }

    class JarReader implements Alloy.Reader {
        private final String _currentPath;
        private final JarFile _jarFile;

        /**
         * Returns a JarReader set at a certain path in the JarFile.
         *
         * @param path    the directory level to be at.
         * @param jarFile the actual jarFile we're reading.
         */
        public JarReader(String path, JarFile jarFile) {
            _currentPath = path;
            _jarFile = jarFile;
        }

        /**
         * Returns a reader that will read resources from that path.
         *
         * @param namespace
         * @return
         * @throws IOException
         */
        @Override public Reader within(String namespace) throws IOException {
            String namespacePath = _currentPath + namespace + "/";
            return new JarReader(namespacePath, _jarFile);
        }

        /**
         * Returns a stream to be able to read from for that resource.
         *
         * @param resourceName
         * @return
         * @throws IOException
         */
        @Override public DataInputStream resource(String resourceName) throws IOException {
            JarEntry je = new JarEntry(_currentPath + resourceName);
            return new DataInputStream(_jarFile.getInputStream(je));
        }
    }


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
    private class IdibonJarDataOutputStream extends DataOutputStream {

        private final JarOutputStream _jos;
        private final ByteArrayOutputStream _baos;
        private final JarEntry _je;
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
}

