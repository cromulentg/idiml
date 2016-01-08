package com.idibon.ml.alloy;

import java.io.*;
import java.util.logging.Logger;

/**
 * A basic shell created for the purpose of writing tests
 *
 * @author Michelle Casbon <michelle@idibon.com>
 */
public class IntentAlloy implements Alloy {
    private static Logger logger = Logger.getLogger(IntentAlloy.class.getName());
    private String _path;
    private double _random;

    public IntentAlloy() {
        this("/tmp");
    }

    public IntentAlloy(String path) {
        _path = path;
        _random = Math.random();
        logger.info("IA: _path:" + _path + " ; random=" + _random);
    }

    public Alloy.Reader reader() {
        return new IntentReader(_path, _random);
    }

    class IntentReader implements Alloy.Reader {
        IntentReader(String path, double random) {
            _path = path;
            _random = random;
            logger.info("IR: _path:" + _path + " ; random=" + _random);
        }

        public DataInputStream resource(String resourceName) throws IOException {
            String filename = _path + "/" + _random + "_" + resourceName;
            FileInputStream fis = new FileInputStream(filename);
            return new DataInputStream(fis);
        }

        public Alloy.Reader within(String namespace) throws IOException {
            return new IntentReader(_path + "/" + namespace, _random);
        }

        private final String _path;
    }

    public Alloy.Writer writer() {
        return new IntentWriter(_path, _random);
    }

    class IntentWriter implements Alloy.Writer {
        IntentWriter(String path, double random) {
            _path = path;
            _random = random;
            logger.info("IW: _path:" + _path + " ; random=" + _random);
        }

        public DataOutputStream resource(String resourceName) throws IOException {
            File dir = new File(_path + "/");
            dir.mkdirs();
            String filename = _path + "/" + _random + "_" + resourceName;
            FileOutputStream fos = new FileOutputStream(filename);
            return new DataOutputStream(fos);
        }

        public Alloy.Writer within(String namespace) throws IOException {
            return new IntentWriter(_path + "/" + namespace, _random);
        }

        private final String _path;
        private final double _random;
    }
}
