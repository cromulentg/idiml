package com.idibon.ml.app

import com.idibon.ml.alloy.JarAlloy
import com.idibon.ml.predict._

import com.typesafe.scalalogging.StrictLogging
import java.io.File
import org.json4s._
import org.json4s.native.JsonMethods._
import scala.collection.JavaConverters._
import scala.io.Source

/** Command-line batch prediction tool with DataFrames
  *
  * Given an alloy and a file containing a list of newline-delimited
  * Idibon JSON documents, predicts all of them and writes the
  * results to an output file
  */
object SparkBatchPredict extends Tool with StrictLogging {

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

    // Do json parsing first, since the libraries are not Serializable
    val docJson = scala.collection.mutable.ListBuffer.empty[JObject]
    Source.fromFile(cli.getOptionValue('i'), "UTF-8")
      .getLines
      .foreach(line => {
        docJson += parse(line).extract[JObject]
      })

    /* split the document into as many partitions as there are CPU threads,
     * to saturate all of the spark context workers */
    val cpuThreads = Runtime.getRuntime().availableProcessors()
    val docJsonRDD = engine.sparkContext.parallelize(docJson, cpuThreads)

    val alloyName = cli.getOptionValue('a')
    val includeFeatures = !cli.hasOption('f')

    val output = new org.apache.commons.csv.CSVPrinter(
      new java.io.PrintWriter(cli.getOptionValue('o'), "UTF-8"),
      org.apache.commons.csv.CSVFormat.RFC4180)

    output.printRecord(Seq("Content", "Label", "Probability").asJava)

    val start = System.currentTimeMillis()
    val processed = new java.util.concurrent.atomic.AtomicLong(0)

    engine.sparkContext.runJob(docJsonRDD, (partition: Iterator[JObject]) => {
      val taskEngine = new com.idibon.ml.common.EmbeddedEngine
      val alloy = JarAlloy.load[Classification](taskEngine, new File(alloyName), false)
      val predictOptionsBuilder = new PredictOptionsBuilder
      if (includeFeatures) predictOptionsBuilder.showSignificantFeatures(0.1f)
      val options = predictOptionsBuilder.build

      partition.map(content => {
        val topPrediction = alloy.predict(content, options).asScala.maxBy(_.probability)
        (content, topPrediction.label, topPrediction.probability)
      }).toList
    }, (partitionId: Int, results: List[(JObject, String, Float)]) => {
      logger.info(s"Partition ${partitionId} processed ${results.size} items")
      processed.addAndGet(results.size)
      results.foreach({ case (content, label, probability) => {
        output.printRecord(Seq(content, label, probability).asJava)
      }})
    })

    val elapsed = System.currentTimeMillis() - start
    logger.info(s"Processed ${processed.get} items in ${elapsed / 1000.0f}s")

    output.close()
  }
}
