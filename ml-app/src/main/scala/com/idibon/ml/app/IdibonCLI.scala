package com.idibon.ml.app

import org.apache.commons.cli.{CommandLine, CommandLineParser, BasicParser, Options}

/** IdibonCLI
  *
  * Provides a set of command-line parsing tools specific to idiml
  *
  */
class IdibonCLI(val argv: Array[String]) {
  private val options = new Options()
  initialize

  private val parser: CommandLineParser = new BasicParser()

  private val cmd: CommandLine = parser.parse(options, argv)

  private val inputFilePath = cmd.getOptionValue("i")

  /** Returns the given input file path.
    *
    * @return the path provided with the -i option. If no path was given, returns a default location
    */
  def getInputFilePath = {
    inputFilePath match {
      case null | "" => "/tmp/idiml.txt"
      case _ => inputFilePath
    }
  }

  private val modelStoragePath = cmd.getOptionValue("m")

  /** Returns the given model storage path.
    *
    * @return the path provided with the -m option. If no path was given, returns a default location
    */
  def getModelStoragePath = {
    modelStoragePath match {
      case null | "" => "/tmp/idiml/mllib_models"
      case _ => modelStoragePath
    }
  }

  /** Defines valid command-line parameters */
  private def initialize = {
    options.addOption("i", true, "input file path")
    options.addOption("m", true, "model storage path")
  }
}
