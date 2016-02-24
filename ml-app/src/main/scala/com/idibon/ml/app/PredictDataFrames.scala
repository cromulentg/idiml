package com.idibon.ml.app

import com.idibon.ml.alloy.JarAlloy
import com.idibon.ml.predict._

import com.typesafe.scalalogging.StrictLogging
import java.io.File
import java.util.concurrent.LinkedBlockingQueue
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods.{compact, render, parse}
import scala.io.Source
import scala.collection.JavaConverters._

/** Command-line batch prediction tool with DataFrames
  *
  * Given an alloy and a file containing a list of newline-delimited
  * Idibon JSON documents, predicts all of them and writes the
  * results to an output file
  */
object PredictDataFrames extends Tool with StrictLogging {

  private [this] def parseCommandLine(argv: Array[String]) = {
    val options = (new org.apache.commons.cli.Options)
      .addOption("i", "input", true, "Input file containing documents.")
      .addOption("o", "output", true, "Result output CSV file.")
      .addOption("a", "alloy", true, "Input alloy.")
      .addOption("f", "no-features", false, "Run without significant features.")
      .addOption("v", "no-validation", false, "Whether we should not validate the model.")

    (new org.apache.commons.cli.BasicParser).parse(options, argv)
  }

  def run(engine: com.idibon.ml.common.Engine, argv: Array[String]) {
    implicit val formats = org.json4s.DefaultFormats

    val cli = parseCommandLine(argv)

    /* split the document into as many partitions as there are CPU threads,
     * to saturate all of the spark context workers */
    val cpuThreads = Runtime.getRuntime().availableProcessors()
    val docText = engine.sparkContext.textFile(cli.getOptionValue('i'), cpuThreads)

    val alloyName = cli.getOptionValue('a')
    val includeFeatures = !cli.hasOption('f')


    val output = new org.apache.commons.csv.CSVPrinter(
      new java.io.PrintWriter(cli.getOptionValue('o'), "UTF-8"),
      org.apache.commons.csv.CSVFormat.RFC4180)

    output.printRecord(Seq("Content", "Label", "Probability").asJava)

    engine.sparkContext.runJob(docText, (partition: Iterator[String]) => {
      val taskEngine = new com.idibon.ml.common.EmbeddedEngine
      val alloy = JarAlloy.load[Classification](taskEngine, new File(alloyName), false)
      val predictOptionsBuilder = new PredictOptionsBuilder
      if (includeFeatures) predictOptionsBuilder.showSignificantFeatures(0.1f)
      val options = predictOptionsBuilder.build

      partition.map(content => {
        val doc = JObject(List(JField("content", JString(content))))
        val topPrediction = alloy.predict(doc, options).asScala.maxBy(_.probability)
        (content, topPrediction.label, topPrediction.probability)
      }).toList
    }, (partitionId: Int, results: List[(String, String, Float)]) => {
      logger.info(s"Partition ${partitionId} processed ${results.size} items")
      results.foreach({ case (content, label, probability) => {
        output.printRecord(Seq(content, label, probability).asJava)
      }})
    })

    output.close()
  }
}
