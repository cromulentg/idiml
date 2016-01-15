package com.idibon.ml.app

class App {
}
object App {
  def main(argv: Array[String]) = {
    val cli : IdibonCLI = new IdibonCLI(argv)
    new com.idibon.ml.predict.EmbeddedEngine().start()

    new com.idibon.ml.train.EmbeddedEngine().start(cli.getInputFilePath, cli.getModelStoragePath)
  }
}
