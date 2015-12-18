package com.idibon.ml.alloy;

import java.io.*;

/**
 * A basic shell created for the purpose of writing tests
 *
 * @author Michelle Casbon <michelle@idibon.com>
 */
public class IntentAlloy implements Alloy {

    public class IntentReader implements Alloy.Reader {
        public DataInputStream resource(String resourceName) throws IOException {
            String filename = "/tmp/" + resourceName + ".txt";
            FileInputStream fis = new FileInputStream(filename);
            return new DataInputStream(fis);
        }

        public IntentReader within(String namespace) throws IOException {
            return this;
        }
    }

    public class IntentWriter implements Alloy.Writer {
        public DataOutputStream resource(String resourceName) throws IOException {
            String filename = "/tmp/" + resourceName + ".txt";
            FileOutputStream fos = new FileOutputStream(filename);
            return new DataOutputStream(fos);
        }

        public IntentWriter within(String namespace) throws IOException {
            return this;
        }
    }

    public IntentReader reader;
    public IntentWriter writer;

    public IntentAlloy() {
        reader = new IntentReader();
        writer = new IntentWriter();
    }
}
