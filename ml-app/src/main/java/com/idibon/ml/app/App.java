package com.idibon.ml.app;

public class App {

    public static void main(String... argv) {
        new com.idibon.ml.predict.EmbeddedEngine().start();
        new com.idibon.ml.train.EmbeddedEngine().start();
    }
}
