package com.idibon.ml.app;

import com.idibon.ml.app.IdibonCLI;

public class App {

    public static void main(String... argv) {
        IdibonCLI cli = new IdibonCLI(argv);
        new com.idibon.ml.predict.EmbeddedEngine().start();
        new com.idibon.ml.train.EmbeddedEngine().start(cli.getInputFilePath(), cli.getModelStoragePath());
    }
}
