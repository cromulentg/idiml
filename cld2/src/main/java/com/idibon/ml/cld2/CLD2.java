package com.idibon.ml.cld2;

import java.nio.charset.Charset;
import java.util.NoSuchElementException;
import java.util.Properties;
import java.nio.file.Files;
import java.nio.file.AccessDeniedException;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.File;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

        // CLD2 only operates on UTF-8 bytes. convert to UTF-8 before calling
        byte[] contentBytes = content.getBytes(UTF_8);

        // Add null terminator character, so we don't get segmentation fault inside of CLD2
        byte[] nullTerminatedContentBytes = new byte[contentBytes.length + 1];
        System.arraycopy(contentBytes, 0, nullTerminatedContentBytes, 0, contentBytes.length);
        nullTerminatedContentBytes[contentBytes.length] = 0;

        int cld = cld2_Detect(nullTerminatedContentBytes, mode == DocumentMode.PlainText);
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

    private static final Logger LOGGER = LoggerFactory.getLogger(CLD2.class);

    /**
     * Extract the correct CLD2 shared library from the JAR and load it.
     *
     * Used by the class static initializer.
     */
    private static void initialize() throws Exception {
        Properties props = new Properties();
        // jni.properties maps OS-Arch pairs to the correct shared library file
        try (InputStream in = CLD2.class.getResourceAsStream("/jni.properties")) {
            if (in == null)
                throw new FileNotFoundException("jni.properties");
            props.load(in);
        }

        String key = "";
        String os = System.getProperty("os.name");
        /* map the rich OS string to a simple identifier used for lookups in
         * the property file. for error logging purposes, keep unmapped
         * operating system names in their entirety, preceded by some invalid
         * characters to ensure that they won't be found in jni.properties */
        if (os.equals("Mac OS X"))
            key += "osx";
        else if (os.startsWith("Windows"))
            key += "windows";
        else if (os.equals("Linux"))
            key += "linux";
        else if (os.equals("FreeBSD"))
            key += "freebsd";
        else
            key += "::" + os;

        String arch = System.getProperty("os.arch");
        // Apple and Oracle seem to disagree how to present x86-64
        if (arch.equals("x86_64") || arch.equals("amd64"))
            key += ".amd64";
        else
            key += "::" + arch;

        LOGGER.debug("initializing CLD2 for {} {}", os, arch);

        String resource = props.getProperty(key);
        if (resource == null)
            throw new NoSuchElementException("library " + key);

        try (InputStream in = CLD2.class.getResourceAsStream(resource)) {
            if (in == null)
                throw new FileNotFoundException(resource);
            // copy the shared library into a temp file
            File systemTemp = new File(System.getProperty("java.io.tmpdir"));
            File localTemp = new File(systemTemp, "idiml/cld2/libs");
            localTemp.mkdirs();

            int extensionPos = resource.lastIndexOf(".");
            String extension = resource.substring(extensionPos);
            final File extract = new File(localTemp, "jni-" + key + extension);

            LOGGER.info("loading '{}' from '{}'", resource, extract);
            // clean up the temp file on exit
            Runtime.getRuntime()
                .addShutdownHook(new Thread(() -> { extract.delete(); }));
            Files.copy(in, extract.toPath(), REPLACE_EXISTING);
            System.load(extract.getAbsolutePath());
        }
    }

    static {
        boolean success = false;
        try {
            initialize();
            success = true;
        } catch (NoSuchElementException |
                 FileNotFoundException |
                 UnsatisfiedLinkError |
                 AccessDeniedException ex) {
            /* these are all common enough errors that logging the full
             * stack is more log noise than desirable */
            LOGGER.error("failed to initialize CLD2\n\t{}: {}",
                         ex.getClass().getSimpleName(), ex.getMessage());
        } catch (Exception ex) {
            LOGGER.error("failed to initialize CLD2", ex);
        } finally {
            INITIALIZED = success;
        }
    }
}
