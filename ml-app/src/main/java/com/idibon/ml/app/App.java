package com.idibon.ml.app;

import com.idibon.ml.predict.*;
import com.idibon.ml.train.*;

public class App {

    public static void main(String... argv) {
        new com.idibon.ml.predict.EmbeddedEngine().start();
        new com.idibon.ml.train.EmbeddedEngine().start();
    }
}
