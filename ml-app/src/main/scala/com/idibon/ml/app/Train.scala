package com.idibon.ml.app

import scala.concurrent.Await
import scala.concurrent.duration.Duration
import scala.util.{Try, Success, Failure}
import scala.io.Source

import com.idibon.ml.train.alloy._
import com.idibon.ml.common.Engine
import com.idibon.ml.predict._
import com.idibon.ml.train.TrainOptions
import com.idibon.ml.alloy.{Alloy, JarAlloy}
import com.idibon.ml.feature.{Buildable, Builder}

import org.json4s._
import org.json4s.native.Serialization.{writePretty => pretty}

import com.typesafe.scalalogging.StrictLogging

import org.json4s.native.JsonMethods
import org.apache.commons.cli

/** Simple command-line trainer tool
  *
  * Given a file containing aggregated training data, generates an
  * alloy and saves the alloy to an output file.
  */
object Train extends Tool with StrictLogging {

  type TrainingData = () => TraversableOnce[JObject]

  implicit val jsonFormats = org.json4s.DefaultFormats

  def run(engine: Engine, argv: Array[String]) {

    val args = Try(new cli.BasicParser().parse(toolOptions, argv)) match {
      case Failure(reason) => {
        logger.error(s"Unable to parse commandline: ${reason.getMessage}")
        null
      }
      case Success(cli) => cli
    }

    if (args == null || args.hasOption("h")) {
      new cli.HelpFormatter().printHelp("ml-app train [options]", toolOptions)
      return
    }

    val name = alloyName(args)
    val alloyConfigJson = readJsonFile(args.getOptionValue('r'))
    val trainerConfigJson = readJsonFile(args.getOptionValue('c'))

    logger.info(s"Training $name with config\n${pretty(trainerConfigJson)}")

    val start = System.currentTimeMillis()
    val easterEgg = if (args.hasOption('w')) Some(new WiggleWiggle()) else None
    val input = if (args.hasOption('i')) Some(args.getOptionValue('i')) else None
    try {
      easterEgg.map(egg => new Thread(egg).start())

      // dispatch
      val alloy = dispatchToTrainer(engine, name,
        alloyConfigJson, trainerConfigJson, input)
      JarAlloy.save(alloy, new java.io.File(args.getOptionValue('o')))
      val elapsed = System.currentTimeMillis() - start
      logger.info(s"Trained $name in ${elapsed / 1000.0}s")
    } finally {
      easterEgg.map(_.terminate())
    }
  }

  /** Dispatches to the appropriate alloy trainer backend
    *
    * @param alloyName name for the new alloy
    * @param alloyConfigJson task / label / rules configuration
    * @param trainerConfigJson configuration data for the furnaces
    * @param trainingData if present, the file containing training data
    */
  private[this] def dispatchToTrainer(engine: Engine,
    alloyName: String, alloyConfigJson: JObject,
    trainerConfigJson: JObject, trainingData: Option[String]): Alloy[_] = {

    val alloyConfig = alloyConfigJson.extract[AlloyConfiguration]
    val forgeConfig = trainerConfigJson.extract[ForgeConfiguration]

    // training data is loaded lazily (and possibly multiple times)
    val readTrainingData = trainingData.map(filename => () => {
      Source.fromFile(filename)
        .getLines
        .map(line => JsonMethods.parse(line).extract[JObject])
    })
    //TODO: check config version for which alloy trainer/forge to use.
    alloyConfig.task_type match {
      case "extraction.bio_ner" =>
        alloyTrainer2(engine, alloyName, alloyConfig,
          forgeConfig, readTrainingData)
      case AlloyTrainer.DOCUMENT_MUTUALLY_EXCLUSIVE |
          AlloyTrainer.DOCUMENT_MULTI_LABEL =>
        alloyTrainer(engine, alloyName, alloyConfigJson,
          trainerConfigJson, readTrainingData)
      case _ =>
        throw new UnsupportedOperationException(s"${alloyConfig.task_type}")
    }
  }

  /** Reads a text file as a single contained JSON object
    *
    * @param filename name of file to read
    * @return parsed JSON object from the file
    */
  private [this] def readJsonFile(filename: String): JObject = {
    val source = Source.fromFile(filename)
    val text = try source.mkString finally source.close()
    JsonMethods.parse(text).extract[JObject]
  }

  /** Returns the name for this alloy, or a default name
    *
    * @param args parsed command line options
    * @return name for the alloy
    */
  private[this] def alloyName(args: cli.CommandLine): String = {
    if (args.hasOption('n'))
      args.getOptionValue('n')
    else
      s"${java.net.InetAddress.getLocalHost.getHostName}-${System.currentTimeMillis}"
  }

  /** Trains an alloy with AlloyTrainer2
    *
    * @param engine current engine context
    * @param alloyName name for the alloy
    * @param alloyConfig parsed alloy configuration data (task type, rules, ...)
    * @param forgeConfig trainer and furnace configuration data
    * @param readTrainingData function to load training data
    */
  private[this] def alloyTrainer2(engine: Engine, alloyName: String,
                                  alloyConfig: AlloyConfiguration, forgeConfig: ForgeConfiguration,
                                  readTrainingData: Option[TrainingData]): Alloy[_] = {

    val labels = alloyConfig.uuid_to_label.map({ case (uuid, name) =>
      new Label(uuid, name)
    })

    val options = TrainOptions()
    readTrainingData.map(f => options.addDocuments(f()))

    val trainer = alloyConfig.task_type match {
      case "extraction.bio_ner" => AlloyForge[Span](engine, forgeConfig.forgeName,
        alloyName, labels.toSeq, forgeConfig.forgeConfig)
    }
    val evaluator = alloyConfig.task_type match {
      case "extraction.bio_ner" => trainer.getEvaluator(engine, alloyConfig.task_type)
      case _ => new NoOpEvaluator()
    }
    Await.result(trainer.forge(options.build(labels.toSeq), evaluator), Duration.Inf)
  }

  /** Trains alloys using the legacy AlloyTrainer mechanism
    *
    * @param engine current engine context
    * @param alloyName name for the alloy
    * @param alloyConfigJson alloy configuration data (task type, rules, ...)
    * @param trainerConfigJson trainer and furnace configuration data
    * @param readTrainingData function to load training data
    */
  private[this] def alloyTrainer(engine: Engine, alloyName: String,
    alloyConfigJson: JObject, trainerConfigJson: JObject,
    readTrainingData: Option[TrainingData]): Alloy[_] = {

    val trainerConfig = trainerConfigJson.extract[ForgeConfiguration]
    val trainer = AlloyFactory.getTrainer(engine, trainerConfig.trainerConfig)

    trainer.trainAlloy(alloyName,
      readTrainingData.getOrElse(() => Seq[JObject]()),
      alloyConfigJson, Some(trainerConfigJson))
  }

  val toolOptions: cli.Options = {
    val options = new cli.Options()

    var opt = new cli.Option("c", "config", true, "JSON Config file for creating a trainer.")
    opt.setRequired(true)
    options.addOption(opt)

    opt = new cli.Option("i", "input", true, "Input file with training data")
    options.addOption(opt)

    opt = new cli.Option("n", "name", true, "Name for the new alloy")
    options.addOption(opt)

    opt = new cli.Option("o", "output", true, "Output file for the alloy")
    opt.setRequired(true)
    options.addOption(opt)

    opt = new cli.Option("r", "label-rules", true, "Input file with label and rules data")
    opt.setRequired(true)
    options.addOption(opt)

    opt = new cli.Option("h", "help", false, "Show help screen")
    options.addOption(opt)

    opt = new cli.Option("w", "Shimmy party!")
    options.addOption(opt)

    options
  }

}

// === JSON schema ====

/** Schema for for label_and_rules (i.e., task / alloy) configuration file
  *
  * @param task_type type of task being trained (classification vs extraction)
  * @param uuid_to_label mapping of UUID strings to the (current) label name
  * @param rules opaque rules data
  */
case class AlloyConfiguration(
  task_type: String,
  uuid_to_label: Map[String, String],
  rules: Option[JObject])

/** Schema for JSON training configuration file */
case class ForgeConfiguration(forgeName: String = "", forgeConfig: JObject = null, trainerConfig: JObject = null, configVersion: String = "0.0.1")
