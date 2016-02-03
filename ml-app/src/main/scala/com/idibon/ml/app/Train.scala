package com.idibon.ml.app

import com.idibon.ml.train.alloy.KClass1FP
import com.idibon.ml.train.datagenerator.SparkDataGeneratorFactory
import com.idibon.ml.train.furnace._

import scala.io.Source
import scala.util.Failure
import org.json4s._

import com.typesafe.scalalogging.StrictLogging

import org.json4s.native.JsonMethods.parse

/** Simple command-line trainer tool
  *
  * Given a file containing aggregated training data, generates an
  * alloy and saves the alloy to an output file.
  */
object Train extends Tool with StrictLogging {

  private [this] def parseCommandLine(argv: Array[String]) = {
    val options = (new org.apache.commons.cli.Options)
      .addOption("i", "input", true, "Input file with training data")
      .addOption("o", "output", true, "Output alloy file")
      .addOption("r", "rules", true, "Input file with rules data")
      .addOption("w", "wiggle-wiggle", false, "Wiggle Wiggle")
      .addOption("n", "ngram", true, "Maximum n-gram size")

    new (org.apache.commons.cli.BasicParser).parse(options, argv)
  }

  def run(engine: com.idibon.ml.common.Engine, argv: Array[String]) {
    implicit val formats = org.json4s.DefaultFormats

    val cli = parseCommandLine(argv)
    val easterEgg = if (cli.hasOption('w')) Some(new WiggleWiggle()) else None
    easterEgg.map(egg => new Thread(egg).start)

    val dataGeneratorConfig = """{"jsonClass":"KClassDataFrameGeneratorBuilder"}"""
    val dataGenerator = SparkDataGeneratorFactory.getDataGenerator(dataGeneratorConfig)
    val furnaceConfig = """{"jsonClass":"XValLogisticRegressionBuilder", "maxIterations":100}"""
    val furnace = FurnaceFactory.getFurnace(engine, furnaceConfig)
    try{
      val startTime = System.currentTimeMillis()
      // default to tri-grams
      val ngramSize = Integer.valueOf(cli.getOptionValue('n', "3")).toInt
      new KClass1FP(engine, dataGenerator, furnace).trainAlloy(
        () => { // training data
          Source.fromFile(cli.getOptionValue('i'))
            .getLines.map(line => parse(line).extract[JObject])
        },
        () => { // rule data
          if (cli.hasOption('r')) {
            Source.fromFile(cli.getOptionValue('r'))
              .getLines.map(line => parse(line).extract[JObject])
          } else {
            List()
          }
        },
        Some(JObject(List(JField("pipelineConfig", JObject(List(JField("ngram", JInt(ngramSize)))))))) // option config
      ).map(alloy => alloy.save(cli.getOptionValue('o')))
        .map(x => {
          val elapsed = System.currentTimeMillis - startTime
          logger.info(s"Training completed in $elapsed ms")
        })
        .recoverWith({ case (error) => {
          logger.error("Unable to train model", error)
          Failure(error)
        }})
    } finally {
      easterEgg.map(_.terminate)
    }

  }

}
