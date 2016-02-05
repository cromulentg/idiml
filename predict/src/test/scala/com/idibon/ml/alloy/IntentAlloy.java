package com.idibon.ml.alloy;

import java.io.*;
import java.util.List;

import com.idibon.ml.predict.PredictOptions;
import com.idibon.ml.predict.Classification;
import org.json4s.JsonAST;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A basic shell created for the purpose of writing tests
 *
 * @author Michelle Casbon <michelle@idibon.com>
 */
public class IntentAlloy implements Alloy<Classification> {
    private static Logger logger = LoggerFactory.getLogger(IntentAlloy.class);
    private String _path;
    private double _random;

    public IntentAlloy() {
        this("/tmp");
    }

    public IntentAlloy(String path) {
        _path = path;
        _random = Math.random();
        logger.info("IA: _path: {} ; random= {}", _path, _random);
    }

    public Alloy.Reader reader() {
        return new IntentReader(_path, _random);
    }

    @Override
    public List<Classification> predict(JsonAST.JObject document, PredictOptions options) {
        return java.util.Collections.emptyList();
    }

    @Override public void save(String path) throws IOException {

    }

    @Override public boolean validate() throws ValidationError {
        return true;
    }

    class IntentReader implements Alloy.Reader {
        IntentReader(String path, double random) {
            _path = path;
            _random = random;
            logger.info("IR: _path: {} ; random= {}", _path, _random);
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
            logger.info("IW: _path: {} ; random= {}", _path, _random);
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
