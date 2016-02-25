package com.idibon.ml.app

import com.idibon.ml.alloy.{Alloy, JarAlloy}
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
      .addOption("b", "benchmark", false, "Run in benchmark mode")
      .addOption("l", "loop", true, "Loop over data set N times")
      .addOption("f", "no-features", false, "Run without significant features.")
      .addOption("v", "no-validation", false, "Whether we should not validate the model.")

    (new org.apache.commons.cli.BasicParser).parse(options, argv)
  }

  /** Run a batch of documents through the alloy to pre-warm all of the
    * code caches for benchmark mode */
  def prewarmJit(engine: com.idibon.ml.common.Engine,
      cli: org.apache.commons.cli.CommandLine) {

    logger.info("== Benchmark Mode == Warming JIT cache")

    // read the first document from the file
    val documents = Source.fromFile(cli.getOptionValue('i'), "UTF-8")
      .getLines.take(1).map(line => parse(line).asInstanceOf[JObject])

    val alloy = JarAlloy.load[Classification](engine,
      new File(cli.getOptionValue('a')), !cli.hasOption('v'))

    val builder = new PredictOptionsBuilder
    if (!cli.hasOption('v')) builder.showSignificantFeatures(0.1f)
    val predictOptions = builder.build

    /* run the document through the prediction loop a bunch of times
     * to make sure that hot loops have a chance for JIT optimization */
    val start = System.currentTimeMillis()
    while (System.currentTimeMillis() - start < 30000) {
      documents.foreach(doc => alloy.predict(doc, predictOptions))
    }
  }

  def run(engine: com.idibon.ml.common.Engine, argv: Array[String]) {
    val cli = parseCommandLine(argv)
    if (cli.hasOption('b')) prewarmJit(engine, cli)

    /* split the document into as many partitions as there are CPU threads,
     * to saturate all of the spark context workers. */
    val cpuThreads = Runtime.getRuntime().availableProcessors()

    val loops = Math.max(1, Integer.parseInt(cli.getOptionValue('l', "1")))

    val includeFeatures = !cli.hasOption('f')

    val output = new org.apache.commons.csv.CSVPrinter(
      new java.io.PrintWriter(cli.getOptionValue('o'), "UTF-8"),
      org.apache.commons.csv.CSVFormat.RFC4180)

    output.printRecord(Seq("Name", "Label", "Probability").asJava)
    ProcessingLoop.loadAlloy(engine, cli.getOptionValue('a'))

    val start = System.currentTimeMillis()

    (1 to loops).foreach(_ => {
      ProcessingLoop.run(engine, cli.getOptionValue('i'),
        cli.getOptionValue('a'), includeFeatures, output)
    })

    output.close()
    val elapsed = System.currentTimeMillis() - start
    logger.info(s"Processed ${ProcessingLoop.processed.get} items in ${elapsed / 1000.0f}s")
  }
}

/** Spark job that executes in the main program loop
  *
  * Broken out to a separate object so that the task is serializable for Spark.
  */
object ProcessingLoop {

  val cpuThreads = Runtime.getRuntime().availableProcessors()
  val processed = new java.util.concurrent.atomic.AtomicLong(0)
  /* this is a poor-man's way of simulating Spark streaming, that only works on
   * localhost-clusters. basically, we prevent serializing this field; however,
   * because the "serialized" function runs within the same JVM as the client,
   * and the localhost driver doesn't actually deserialize the transferred
   * spark job, the alloy reference is ultimately shared across workers */
  @transient var alloy: Alloy[Classification] = null

  def loadAlloy(engine: com.idibon.ml.common.Engine, alloyFilename: String) {
    this.alloy = JarAlloy.load[Classification](engine, new File(alloyFilename), false)
  }

  def run(engine: com.idibon.ml.common.Engine, documentFilename: String,
      alloyFilename: String, includeFeatures: Boolean,
      output: org.apache.commons.csv.CSVPrinter) {

    val docRDD = engine.sparkContext.textFile(documentFilename, cpuThreads)
    engine.sparkContext.runJob(docRDD, (partition: Iterator[String]) => {
      val predictOptionsBuilder = new PredictOptionsBuilder
      if (includeFeatures) predictOptionsBuilder.showSignificantFeatures(0.1f)
      val options = predictOptionsBuilder.build

      partition.map(content => {
        val doc = parse(content).asInstanceOf[JObject]
        val name = (doc \ "name").asInstanceOf[JString].s
        val topPrediction = this.alloy.predict(doc, options).asScala.maxBy(_.probability)
        (name, topPrediction.label, topPrediction.probability)
      }).toList
    }, (partitionId: Int, results: List[(String, String, Float)]) => {
      processed.addAndGet(results.size)
      results.foreach({ case (content, label, probability) => {
        output.printRecord(Seq(content, label, probability).asJava)
      }})
    })
  }
}
