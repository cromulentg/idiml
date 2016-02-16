package com.idibon.ml.app

import java.io.File

import com.idibon.ml.alloy.JarAlloy
import com.typesafe.scalalogging.StrictLogging

/** Command-line to get training data from an alloy.
  *
  * Given an alloy, writes training data to output
  */
object GetTrainingStats extends Tool with StrictLogging {

  private [this] def parseCommandLine(argv: Array[String]) = {
    val options = (new org.apache.commons.cli.Options)
      .addOption("o", "output", true, "directory to output data to.")
      .addOption("p", "prefix", true, "prefix to give files in directory.")
      .addOption("a", "alloy", true, "Input alloy.")

    (new org.apache.commons.cli.BasicParser).parse(options, argv)
  }

  def run(engine: com.idibon.ml.common.Engine, argv: Array[String]) {
    implicit val formats = org.json4s.DefaultFormats

    val cli = parseCommandLine(argv)
    val summaries = JarAlloy.getTrainingSummaries(new File(cli.getOptionValue('a')))
    summaries.foreach(println)
    // TODO: save to some sort of file for ingestion into plotting etc.
  }
}
