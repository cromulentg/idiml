package com.idibon.ml.cld2;

import java.nio.charset.Charset;
import java.util.NoSuchElementException;
import java.util.Properties;
import java.nio.file.Files;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.File;

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;

/**
 * JNI wrapper around the Compact Language Detector 2 (CLD2) library,
 * available at https://github.com/CLD2Owners/cld2
 */
public final class CLD2 {

    public enum DocumentMode {
        PlainText,
        HTML
    };

    /**
     * Performs language detection on a string.
     *
     * @param content the string to detect
     * @param mode which content extraction mode to use for language detection
     * @return the document's primary language identifer
     */
    public static LangID detect(String content, DocumentMode mode) {
        mustBeInitialized();
        int cld = cld2_Detect(content.getBytes(UTF_8),
                              mode == DocumentMode.PlainText);
        return LangID.of(Math.abs(cld));
    }

    /**
     * Throws an IllegalStateException if the JNI initialization failed
     */
    private static void mustBeInitialized() {
        if (!INITIALIZED)
            throw new IllegalStateException("Failed to initialize CLD2");
    }

    /**
     * C calling convention wrapper to C++ CLD detection function
     *
     * Returns the language ID if the detection was reliable, or the
     * negative language ID if unreliable.
     *
     * @param content the content to detect, must be valid UTF-8
     * @param plainText when true, the document is treated as plain text;
     *    when false, the document is treated as HTML and tags are ignored
     * @return the value of the langauge ID
     */
    private static native int cld2_Detect(byte[] content, boolean plainText);

    private static final Charset UTF_8 = Charset.forName("UTF-8");

    private static final boolean INITIALIZED;

    private static void initialize() throws Exception {
        Properties props = new Properties();
        try (InputStream in = CLD2.class.getResourceAsStream("/jni.properties")) {
            if (in == null)
                throw new FileNotFoundException("jni.properties");
            props.load(in);
        }

        String key = "";
        String os = System.getProperty("os.name");
        if (os.equals("Mac OS X"))
            key += "osx";
        else if (os.startsWith("Windows"))
            key += "windows";
        else if (os.equals("Linux"))
            key += "linux";
        else if (os.equals("FreeBSD"))
            key += "freebsd";

        String arch = System.getProperty("os.arch");
        if (arch.equals("x86_64") || arch.equals("amd64"))
            key += ".amd64";

        String resource = props.getProperty(key);
        if (resource == null)
            throw new NoSuchElementException("library " + key);

        try (InputStream in = CLD2.class.getResourceAsStream(resource)) {
            if (in == null)
                throw new FileNotFoundException(resource);
            File systemTemp = new File(System.getProperty("java.io.tmpdir"));
            File localTemp = new File(systemTemp, "idiml/cld2/libs");
            localTemp.mkdirs();

            int extensionPos = resource.lastIndexOf(".");
            String extension = resource.substring(extensionPos);
            final File extract = new File(localTemp, "jni-" + key + extension);
            // clean up the temp file on exit
            Runtime.getRuntime()
                .addShutdownHook(new Thread(() -> { extract.delete(); }));
            Files.copy(in, extract.toPath(), REPLACE_EXISTING);
            System.load(extract.getAbsolutePath());
        }
    }

    /* Dynamically detect the correct library to load, extract it from
     * the JAR, and load it */
    static {
        boolean success = false;
        try {
            initialize();
            success = true;
        } catch (Exception ex) {
            // FIXME: log exception
            System.err.printf("%s\n", ex.getMessage());
        } finally {
            INITIALIZED = success;
        }
    }
}
