package com.idibon.ml.app

import com.typesafe.scalalogging.StrictLogging
import java.io.File
import java.nio.channels.FileChannel

import scala.io.Source
import scala.collection.JavaConverters._
import java.util.concurrent.LinkedBlockingQueue
import org.json4s._
import org.json4s.native.JsonMethods.parse
import com.idibon.ml.alloy.JarAlloy
import com.idibon.ml.predict._
import org.json4s.native.JsonMethods.parse

/** Command-line batch prediction tool
  *
  * Given an alloy and a file containing a list of newline-delimited
  * Idibon JSON documents, predicts all of them and writes the
  * results to an output file
  */
object Predict extends Tool with StrictLogging {

  private [this] def parseCommandLine(argv: Array[String]) = {
    val options = (new org.apache.commons.cli.Options)
      .addOption("i", "input", true, "Input file containing documents.")
      .addOption("o", "output", true, "Result output CSV file.")
      .addOption("a", "alloy", true, "Input alloy.")
      .addOption("b", "benchmark", false, "Run in benchmark mode (prime the JIT")
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

    /* run the document through the prediction loop for 30 seconds
     * to make sure that hot loops get a chance for JIT optimization */
    val start = System.currentTimeMillis()
    while (System.currentTimeMillis() - start < 30000) {
      documents.foreach(doc => alloy.predict(doc, predictOptions))
    }
  }

  def run(engine: com.idibon.ml.common.Engine, argv: Array[String]) {
    implicit val formats = org.json4s.DefaultFormats

    val cli = parseCommandLine(argv)

    if (cli.hasOption('b')) prewarmJit(engine, cli)

    val model = JarAlloy.load[Classification](engine,
      new File(cli.getOptionValue('a')), !cli.hasOption('v'))

    logger.info(s"Loaded Alloy.")
    /* results will be written to the output file by a consumer thread;
     * after the last document is predicted, the main thread will post
     * a sentinel value of None to cause the result output thread to
     * terminate. */
    val results = new LinkedBlockingQueue[Option[(JObject, Seq[Classification])]]
    val output = new org.apache.commons.csv.CSVPrinter(
      new java.io.PrintWriter(cli.getOptionValue('o'), "UTF-8"),
      org.apache.commons.csv.CSVFormat.RFC4180)
    val includeFeatures = !cli.hasOption('f')

    val labels = model.getLabels.asScala.sortWith(_.name < _.name)
    val labelCols = if (includeFeatures) {
      labels.flatMap(l => List(l.name, s"features[${l.name}]"))
    } else {
      labels.map(_.name)
    }

    output.printRecord((Seq("Name", "Content") ++ labelCols).asJava)

    val processed = new java.util.concurrent.atomic.AtomicLong(0)

    val resultsThread = new Thread(new Runnable() {
      override def run {
        Stream.continually(results.take)
          .takeWhile(_.isDefined)
          .foreach(_ match {
            case Some((document, prediction)) => {
              val predictionByLabel = prediction.map(p => (p.label, p)).toMap
              val sortedPredictions = labels.map(l => predictionByLabel.get(l.uuid.toString))
              val outputCols = if (includeFeatures) {
                sortedPredictions.map(_.map(p => {
                  List(p.probability, p.significantFeatures.map(f => {
                    f._1.getAsString match {
                      case Some(feat) => feat
                      case None => ""
                    }
                  }))
                }).getOrElse(List(0.0f, List()))).reduce(_ ++ _)
              } else {
                sortedPredictions.map(_.map(_.probability).getOrElse(0.0f))
              }

              val row = (Seq(
                (document \ "name").extract[Option[String]].getOrElse(""),
                (document \ "content").extract[String]) ++ outputCols).asJava

              processed.incrementAndGet()
              output.printRecord(row)
            }
            case _ => { }
          })
      }
    })
    resultsThread.start

    val loops = Math.max(1, Integer.parseInt(cli.getOptionValue('l', "1")))
    /* in order to eliminate syscall overhead from affecting benchmark
     * performance with (very) small batch sizes, memory-map the file once
     * and just create input streams to wrap the data on each loop */
    val mappedFile = if (loops > 5 && cli.hasOption('b')) {
      val fd = FileChannel.open(new File(cli.getOptionValue('i')).toPath)
      val mapping = fd.map(FileChannel.MapMode.READ_ONLY, 0, fd.size())
      fd.close()
      Some(mapping)
    } else {
      None
    }

    val start = System.currentTimeMillis()

    try {
      val optionsBuilder = new PredictOptionsBuilder
      if (includeFeatures) optionsBuilder.showSignificantFeatures(0.1f)
      val predictOptions = optionsBuilder.build

      (1 to loops).foreach(_ => {
        /* either wrap an input stream (and then a source) around the memory-
         * mapped file, if it exists, or create a source for the file */
        val source = mappedFile.map(mapping => {
          mapping.rewind()
          Source.fromInputStream(new ByteBufferInputStream(mapping), "UTF-8")
        }).getOrElse(Source.fromFile(cli.getOptionValue('i'), "UTF-8"))

        source.getLines.toStream.par
          .foreach(line => {
            val document = parse(line).asInstanceOf[JObject]
            val result = model.predict(document, predictOptions)
            results.offer(Some((document, result.asScala)))
          })
      })
    } finally {
      // send the sentinel value to shut down the output thread
      results.offer(None)
      // wait for the output thread to finish and close the stream
      resultsThread.join
      output.close
      val elapsed = System.currentTimeMillis() - start
      val rate = processed.get / (elapsed / 1000.0f)
      logger.info(s"Processed ${processed.get} items in ${elapsed / 1000.0f}secs. Rate: " + f"$rate%1.9f items/sec")
    }
  }

  /** Simple InputStream for reading from a ByteBuffer */
  private [this] class ByteBufferInputStream(buffer: java.nio.ByteBuffer)
      extends java.io.InputStream {

    override def read: Int = {
      // convert signed bytes to unsigned ints
      if (buffer.hasRemaining)
        (buffer.get + 0x100) & 0xff
      else
        -1
    }

    override def read(b: Array[Byte], off: Int, len: Int) = {
      val consume = Math.min(len, buffer.remaining())
      if (consume > 0) {
        buffer.get(b, off, consume)
        consume
      } else {
        -1
      }
    }

    override def mark(limit: Int) {
      buffer.mark()
    }

    override def reset() {
      buffer.reset()
    }

    override def markSupported = true
  }
}
