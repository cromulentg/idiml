package com.idibon.ml.alloy;

import java.io.*;

/**
 * A basic shell created for the purpose of writing tests
 *
 * @author Michelle Casbon <michelle@idibon.com>
 */
public class IntentAlloy implements Alloy {
    public Alloy.Reader reader() {
        return new IntentReader("/tmp");
    }

    class IntentReader implements Alloy.Reader {
        IntentReader(String path) {
            _path = path;
        }

        public DataInputStream resource(String resourceName) throws IOException {
            String filename = _path + "/" + resourceName;
            FileInputStream fis = new FileInputStream(filename);
            return new DataInputStream(fis);
        }

        public Alloy.Reader within(String namespace) throws IOException {
            return new IntentReader(_path + "/" + namespace);
        }

        private final String _path;
    }

    public Alloy.Writer writer() {
        return new IntentWriter("/tmp");
    }

    class IntentWriter implements Alloy.Writer {
        IntentWriter(String path) {
            _path = path;
        }

        public DataOutputStream resource(String resourceName) throws IOException {
            String filename = _path + "/" + resourceName;
            FileOutputStream fos = new FileOutputStream(filename);
            return new DataOutputStream(fos);
        }

        public Alloy.Writer within(String namespace) throws IOException {
            return new IntentWriter(_path + "/" + namespace);
        }

        private final String _path;
    }
}
