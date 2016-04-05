package com.idibon.ml.app

import scala.collection.JavaConverters._
import scala.util.{Try, Success, Failure}
import scala.io.Source

import com.idibon.ml.common.Engine
import com.idibon.ml.train.TrainOptions
import com.idibon.ml.feature.{Feature, ProductFeature, FeaturePipelineLoader}

import org.json4s._

import com.typesafe.scalalogging.StrictLogging

import org.json4s.native.JsonMethods
import org.apache.commons.cli

/** Command line tool to get feature insights from the passed in data.
  *
  * Given a file containing training data, generates n-gram counts.
  */
object Insights extends Tool with StrictLogging {

  type TrainingData = () => TraversableOnce[JObject]

  implicit val jsonFormats = org.json4s.DefaultFormats

  /**
    * Main function that orchestrates this app.
    *
    * This is a pretty long function that could be separated into smaller pieces
    * quite easily...
    *
    * @param engine - the Idiml Engine context to use
    * @param argv - command-line options to configure tool
    */
  def run(engine: Engine, argv: Array[String]) {
    // parse the command line options
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
    val start = System.currentTimeMillis()
    // get data to iterate over
    val trainingSet = getDataToProcess(args)
    val pipelineConfig = if (args.hasOption("c")) {
      readJSONFile(args.getOptionValue('c'))
    } else {
      readJSONResource("insightsConfigs/base_insights_config.json")
    }

    val fp = new FeaturePipelineLoader().load(engine, None, Some(pipelineConfig))
    // prime the pipeline
    val frozen = fp.prime(trainingSet.dataSet.train())
    // aggregate dimensions
    val summed = new Array[Int](frozen.totalDimensions.get)
    trainingSet.dataSet.train().foreach(d => {
      val vect = frozen.apply(d)
      vect.foreachActive((index, value) => {
        summed(index) += value.toInt
      })
    })
    // sort them into (n-gram length, -count, dimension)
    val sortedCounts = summed.zipWithIndex
      .map({case (count, dimension) => (count, dimension, frozen.getFeatureByIndex(dimension))})
      .sortBy({case (count, dimension, feature) =>
        (-feature.get.asInstanceOf[ProductFeature].features.size, // ngram
          -count, // count
          feature.get.asInstanceOf[ProductFeature].getHumanReadableString.get) // string value
      })

    val baseFileName: String = args.getOptionValue('o')
    // save the sorted results to file
    val output = new org.apache.commons.csv.CSVPrinter(
      new java.io.PrintWriter(baseFileName, "UTF-8"),
      org.apache.commons.csv.CSVFormat.RFC4180)
    output.printRecord(Seq("NGram", "Count", "Value").asJava)
    sortedCounts.foreach({case (count, dimension, feature) =>
        val feat: Feature[_] = feature.get
      output.printRecord(
        Seq(feat.asInstanceOf[ProductFeature].features.size,
          count,
          feat.getHumanReadableString.get).asJava)
    })
    output.close()
    // dispatch
    val elapsed = System.currentTimeMillis() - start
    logger.info(s"Save insights to ${args.getOptionValue('o')} in ${elapsed / 1000.0}s")
  }

  /** Reads a text file as a single contained JSON object
    *
    * @param filename name of file to read
    * @return parsed JSON object from the file
    */
  private [this] def readJSONFile(filename: String): JObject = {
    val source = Source.fromFile(filename)
    val text = try source.mkString finally source.close()
    JsonMethods.parse(text).extract[JObject]
  }

  /** Reads a text resource file as a single contained JSON object
    *
    * @param resourceName name of file to read
    * @return parsed JSON object from the file
    */
  private [this] def readJSONResource(resourceName: String): JObject = {
    val resource = getClass.getClassLoader.getResourceAsStream(resourceName)
    val source = Source.fromInputStream(resource)
    val text = try source.mkString finally source.close()
    JsonMethods.parse(text).extract[JObject]
  }


  /**
    * Gets the data to process.
    *
    * @param args
    * @return
    */
  def getDataToProcess(args: cli.CommandLine): TrainOptions = {
    val input = if (args.hasOption('i')) Some(args.getOptionValue('i')) else None
    val readTrainingData = input.map(filename => () => {
      Source.fromFile(filename)
        .getLines
        .map(line => JsonMethods.parse(line).extract[JObject])
    })
    val options = TrainOptions()
    readTrainingData.map(f => options.addDocuments(f()))
    options.build(Seq())
  }

  val toolOptions: cli.Options = {
    val options = new cli.Options()

    var opt = new cli.Option("c", "config", true, "JSON Config file for creating a the feature pipeline.")
    opt.setRequired(false)
    options.addOption(opt)

    opt = new cli.Option(
      "i", "input", true, "Input file with data to process insights from; i.e. JSON with a content field.")
    opt.setRequired(true)
    options.addOption(opt)

    opt = new cli.Option("o", "output", true, "Output file to write the CSV results to.")
    opt.setRequired(true)
    options.addOption(opt)

    opt = new cli.Option("h", "help", false, "Show help screen")
    options.addOption(opt)

    options
  }
}

