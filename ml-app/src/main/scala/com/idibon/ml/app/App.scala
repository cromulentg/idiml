package com.idibon.ml.app

class App {
}
object App {
  def main(argv: Array[String]) = {
    val cli : IdibonCLI = new IdibonCLI(argv)
    // train model
    new com.idibon.ml.train.EmbeddedEngine().start(cli.getInputFilePath, cli.getModelStoragePath)
    // then load and predict on it
    new com.idibon.ml.predict.EmbeddedEngine().start(cli.getModelStoragePath)
  }
}
