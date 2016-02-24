package com.idibon.ml.app

import com.idibon.ml.common.Engine
import com.idibon.ml.predict.PredictOptionsBuilder
import com.idibon.ml.train.alloy._
import com.typesafe.scalalogging.StrictLogging
import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization._

import scala.collection.JavaConverters._
import scala.io.Source

/**
  * A tool to make it easy to test training and prediction without
  * saving a model as an alloy. This should be used for development/
  * experimentation.
  *
  */
object QuickTrainAndPredict extends Tool with StrictLogging {

  private [this] def parseCommandLine(argv: Array[String]) = {
    val options = (new org.apache.commons.cli.Options)
      .addOption("i", "input", true, "Input file with training data")
      .addOption("o", "output", true, "Output alloy file")
      .addOption("r", "rules", true, "Input file with rules data")
      .addOption("c", "config", false, "JSON Config file for creating a trainer.")

    new (org.apache.commons.cli.BasicParser).parse(options, argv)
  }


  /** Executes the tool
    *
    * @param engine - the Idiml Engine context to use
    * @param argv   - command-line options to configure tool
    */
  override def run(engine: Engine, argv: Array[String]): Unit = {
    implicit val formats = org.json4s.DefaultFormats

    val cli = parseCommandLine(argv)
    val startTime = System.currentTimeMillis()
    // get the config file else the default one
    val configFilePath = if (cli.getOptionValue('c', "").isEmpty()) {
      getClass.getClassLoader.getResource("trainerConfigs/base_kbinary_learning_curves.json").getPath()
    } else cli.getOptionValue('c')
    val trainingJobJValue = parse(Source.fromFile(configFilePath).reader())
    logger.info(s"Reading in Config ${writePretty(trainingJobJValue)}")
    val trainer = AlloyFactory.getTrainer(engine, (trainingJobJValue \ "trainerConfig").extract[JObject])
    val line = Source.fromFile(cli.getOptionValue('r'))
      .getLines().foldLeft(new StringBuilder())((bld, jsn) => bld.append(jsn)).mkString
    val model = trainer.trainAlloy("quickTrain",
      () => { // training data
      Source.fromFile(cli.getOptionValue('i'))
        .getLines.map(line => parse(line).extract[JObject])
      },
        parse(line).extract[JObject]
      , //take feature pipeline from passed in configuration
      Some(trainingJobJValue.extract[JObject])
    )
    val elapsed = System.currentTimeMillis - startTime
    logger.info(s"Training completed in $elapsed ms")

//
//    val doc = new JObject(List("content" -> new JString("this is some content")))
//    val result = model.predict(doc,
//      new PredictOptionsBuilder().showSignificantFeatures(0.01f).build()).asScala
  }
}
