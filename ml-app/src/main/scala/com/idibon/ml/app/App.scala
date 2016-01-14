package com.idibon.ml.app

class App(val argv: Array[String]) {

  def main = {
    val cli : IdibonCLI = new IdibonCLI(argv)
    new com.idibon.ml.predict.EmbeddedEngine().start()

    new com.idibon.ml.train.EmbeddedEngine().start(cli.getInputFilePath, cli.getModelStoragePath)
  }

  main
}
