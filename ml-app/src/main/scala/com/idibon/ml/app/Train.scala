package com.idibon.ml.app

import com.idibon.ml.train.alloy.AlloyFactory
import com.idibon.ml.alloy.JarAlloy

import java.io.File
import scala.io.Source
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
      .addOption("r", "label-rules", true, "Input file with label and rules data. Required.")
      .addOption("w", "wiggle-wiggle", false, "Wiggle Wiggle")
      .addOption("c", "config", true, "JSON Config file for creating a trainer.")
      .addOption("n", "name", true, "Provide a name for this alloy.")

    new (org.apache.commons.cli.BasicParser).parse(options, argv)
  }

  def run(engine: com.idibon.ml.common.Engine, argv: Array[String]) {
    implicit val formats = org.json4s.DefaultFormats
    val cli = parseCommandLine(argv)
    if (!cli.hasOption('r')) throw new IllegalArgumentException("Error: Labels and rules configuration not found.")
    val easterEgg = if (cli.hasOption('w')) Some(new WiggleWiggle()) else None
    easterEgg.map(egg => new Thread(egg).start)
    // get the config file else the default one
    val configFileStream = if (cli.getOptionValue('c', "").isEmpty()) {
      new java.io.InputStreamReader(getClass.getClassLoader
        .getResourceAsStream("trainerConfigs/base_kbinary_xval_config.json"))
    } else {
      new java.io.FileReader(cli.getOptionValue('c'))
    }
    val trainingJob = parse(configFileStream).extract[JObject]
    configFileStream.close
    logger.info(s"Reading in Config ${writePretty(trainingJob)} from ${cli.getOptionValue('c')}")
    val trainer = AlloyFactory.getTrainer(engine, (trainingJob \ "trainerConfig").extract[JObject])

    val rulesFile = Source.fromFile(cli.getOptionValue('r'))
    val labelsAndRules = try parse(rulesFile.mkString).extract[JObject] finally rulesFile.close

    def readDocumentsFn() = {
      Source.fromFile(cli.getOptionValue('i'))
        .getLines.map(line => parse(line).extract[JObject])
    }

    try {
      val startTime = System.currentTimeMillis()
      val name = if (cli.hasOption('n')) {
        cli.getOptionValue('n')
      } else {
        java.net.InetAddress.getLocalHost.getHostName() + "-" + System.currentTimeMillis().toString
      }
      val alloy = trainer.trainAlloy(name,
        readDocumentsFn, labelsAndRules, Some(trainingJob))

      JarAlloy.save(alloy, new File(cli.getOptionValue('o')))
      val elapsed = (System.currentTimeMillis - startTime) / 1000.0
      logger.info(s"Training completed in $elapsed s")
    } catch {
      case e: Throwable => logger.error("Unable to train model", e)
    } finally {
      easterEgg.map(_.terminate)
    }

  }

}
