package com.idibon.ml.app

import com.idibon.ml.train.alloy.AlloyFactory

import scala.io.Source
import scala.util.Failure
import org.json4s._
import org.json4s.native.Serialization.writePretty

import com.typesafe.scalalogging.StrictLogging

import org.json4s.native.JsonMethods.parse

/** Simple command-line trainer tool
  *
  * Given a file containing aggregated training data, generates an
  * alloy and saves the alloy to an output file.
  */
object Train extends Tool with StrictLogging {

  private[this] def parseCommandLine(argv: Array[String]) = {
    val options = (new org.apache.commons.cli.Options)
      .addOption("i", "input", true, "Input file with training data")
      .addOption("o", "output", true, "Output alloy file")
      .addOption("r", "rules", true, "Input file with rules data")
      .addOption("w", "wiggle-wiggle", false, "Wiggle Wiggle")
      .addOption("c", "config", false, "JSON Config file for creating a trainer.")

    new (org.apache.commons.cli.BasicParser).parse(options, argv)
  }

  def run(engine: com.idibon.ml.common.Engine, argv: Array[String]) {
    implicit val formats = org.json4s.DefaultFormats

    val cli = parseCommandLine(argv)
    val easterEgg = if (cli.hasOption('w')) Some(new WiggleWiggle()) else None
    easterEgg.map(egg => new Thread(egg).start)
    // get the config file else the default one
    val configFileStream = if (cli.getOptionValue('c', "").isEmpty()) {
      new java.io.InputStreamReader(getClass.getClassLoader
        .getResourceAsStream("trainerConfigs/base_kbinary_xval_config.json"))
    } else {
      new java.io.FileReader(cli.getOptionValue('c'))
    }
    val trainingJobJValue = parse(configFileStream)
    configFileStream.close
    logger.info(s"Reading in Config ${writePretty(trainingJobJValue)}")
    val trainer = AlloyFactory.getTrainer(engine, (trainingJobJValue \ "trainerConfig").extract[JObject])
    try {
      val startTime = System.currentTimeMillis()
      trainer.trainAlloy(
        () => {
          // training data
          Source.fromFile(cli.getOptionValue('i'))
            .getLines.map(line => parse(line).extract[JObject])
        },
        () => {
          // rule data
          if (cli.hasOption('r')) {
            Source.fromFile(cli.getOptionValue('r'))
              .getLines.map(line => parse(line).extract[JObject])
          } else {
            List()
          }
        },
        Some(trainingJobJValue.extract[JObject])
      ).map(alloy => alloy.save(cli.getOptionValue('o')))
        .map(x => {
          val elapsed = System.currentTimeMillis - startTime
          logger.info(s"Training completed in $elapsed ms")
        })
        .recoverWith({ case (error) => {
          logger.error("Unable to train model", error)
          Failure(error)
        }
        })
    } finally {
      easterEgg.map(_.terminate)
    }

  }

}
