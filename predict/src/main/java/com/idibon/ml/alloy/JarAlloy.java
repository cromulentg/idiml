package com.idibon.ml.alloy;

import java.io.*;
import java.util.jar.*;

/**
 * Rough draft of a Jar backed Alloy.
 * <p>
 * This is very simple, and allows you to write to a jar as if it was a file system.
 * Writing:
 * It doesn't do any signing or anything complex with Manifests.
 * Reading:
 * It doesn't do anything interesting, other than returning streams for reading.
 * <p>
 * Open questions:
 * - Does IdibonJarDataOutputStream suffice?
 * <p>
 * http://docs.oracle.com/javase/8/docs/technotes/guides/jar/jar.html - jar spec
 * https://docs.oracle.com/javase/8/docs/api/java/util/jar/package-summary.html - code
 *
 * @author "Stefan Krawczyk <stefan@idibon.com>"
 */
public class JarAlloy implements Alloy {

    private String _path;
    private JarOutputStream _jos;
    private JarFile _jar;

    public JarAlloy(String pathToJar) {
        _path = pathToJar;
    }

    /**
     * Returns a reader that will read from the root of the JarFile setup when the
     * JarAlloy was instantiated.
     *
     * @return
     * @throws IOException
     */
    public Alloy.Reader reader() throws IOException {
        _jar = new JarFile(new File(_path));
        // start base path off at "root"
        return new JarReader("", _jar);
    }

    class JarReader implements Alloy.Reader {
        String _currentPath;
        JarFile _jarFile;

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
     * @return
     * @throws IOException
     */
    public Alloy.Writer writer() throws IOException {
        //TODO: this manifest stuff & more will probably be passed in...
        Manifest manifest = new Manifest();
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
        _jos = new JarOutputStream(new FileOutputStream(new File(_path)), manifest);
        // start base path off at "root"
        return new JarWriter("");
    }

    class JarWriter implements Alloy.Writer {

        String _currentPath;

        /**
         * Constructor that sets up from what path a Writer will start writing at.
         *
         * @param path
         */
        public JarWriter(String path) {
            _currentPath = path;
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
            return new JarWriter(namespacePath);
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

    /**
     * Closes writing or reading a Jar file if one was being written or read from.
     *
     * @throws IOException
     */
    public void close() throws IOException {
        if (_jos != null)
            _jos.close();
        if (_jar != null)
            _jar.close();
    }


    public static void main(String[] args) {
        try {
            // create alloy to save jar.
            JarAlloy ja = new JarAlloy(args[0]);
            // base writer
            Writer jw = ja.writer();
            // give me a writer that will write to /aFolder/
            Writer jw2 = jw.within("aFolder");
            // write to /aFolder/test.json
            DataOutputStream dos = jw2.resource("test.json");
            Codec.String.write(dos, "{\"this\": \"is some json!\"}");
            // close the resource, aka. the jar entry for test.json
            dos.close();
            // write to test2.json
            dos = jw.resource("test2.json");
            Codec.String.write(dos, "{\"this\": \"is some more json!\"}");
            // close the resource, aka. the jar entry for test2.json
            dos.close();
            // close writing to the JAR.
            ja.close();
            // create reader to read in JAR just created.
            Reader reader = ja.reader();
            // let's read test.json
            DataInputStream stream = reader.resource("test2.json");
            System.out.println(Codec.String.read(stream));
            // close the input stream
            stream.close();
            // let's create a reader to read from /aFolder/
            Reader reader2 = reader.within("aFolder");
            // lets read /aFolder/test.json
            DataInputStream stream2 = reader2.resource("test.json");
            System.out.println(Codec.String.read(stream2));
            // close input stream
            stream2.close();
            // close reading from JarFile
            ja.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
