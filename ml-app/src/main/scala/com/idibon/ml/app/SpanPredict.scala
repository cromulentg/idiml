package com.idibon.ml.app

import java.util.concurrent.LinkedBlockingQueue
import java.io.{File, PrintWriter}
import scala.io.Source
import scala.collection.JavaConverters._
import scala.util.{Try, Success, Failure}

import com.idibon.ml.alloy._
import com.idibon.ml.predict._
import com.idibon.ml.common.Engine

import org.apache.commons.cli
import org.apache.commons.csv
import com.typesafe.scalalogging.StrictLogging
import org.json4s.native.JsonMethods
import org.json4s.JObject

/** Extracts named entities from documents using span prediction alloys
  *
  * Outputs a 6-column CSV for each extracted Span:
  *  1) Document Name 2) Span Text 3) Label 4) Probability 5) Offset 6) Length
  */
object SpanPredict extends Tool with StrictLogging {

  def run(engine: Engine, argv: Array[String]) {
    implicit val formats = org.json4s.DefaultFormats

    val args = Try(new cli.BasicParser().parse(toolOptions, argv)) match {
      case Failure(reason) => {
        logger.error(s"Unable to parse commandline: ${reason.getMessage}")
        null
      }
      case Success(parsed) => parsed
    }

    if (args == null || args.hasOption("h")) {
      new cli.HelpFormatter().printHelp("ml-app predict [options]", toolOptions)
      return
    }

    val alloy = JarAlloy.load[Span](engine,
      new File(args.getOptionValue('a')), args.hasOption('v'))

    val output = new csv.CSVPrinter(
      new PrintWriter(args.getOptionValue('o'), "UTF-8"), csv.CSVFormat.RFC4180)

    output.printRecord("Name Span Label Confidence Offset Length".split(" ").toSeq.asJava)

    logger.info("Loaded alloy")

    // write all prediction results from a consumer thread
    val results = new LinkedBlockingQueue[Option[(JObject, Seq[Span])]]

    val processed = new java.util.concurrent.atomic.AtomicLong(0)
    val resultsThread = new Thread(new Runnable() {
      override def run {
        Stream.continually(results.take)
          .takeWhile(_.isDefined)
          .foreach(result => {
            val (document, predictions) = result.get
            val name = (document \ "name").extract[Option[String]].getOrElse("")
            val content = (document \ "content").extract[String]

            predictions.foreach(span => {
              val label = alloy.translateUUID(span.label)
              val text = content.substring(span.offset, span.end)
              output.printRecord(Seq(name, text, label.name, span.probability,
                span.offset, span.length).asJava)
            })
            processed.incrementAndGet()
          })
      }
    })

    val source = Source.fromFile(args.getOptionValue('i'), "UTF-8")

    resultsThread.start()
    val start = System.currentTimeMillis()
    try {
      source.getLines.toStream.par.foreach(line => {
        val document = JsonMethods.parse(line).extract[JObject]
        val result = alloy.predict(document, PredictOptions.DEFAULT)
        results.offer(Some(document -> result.asScala))
      })
    } finally {
      results.offer(None)
      resultsThread.join()
      output.close()
      val elapsed = (System.currentTimeMillis() - start) / 1000.0
      val rate = processed.get() / elapsed
      logger.info(s"Processed ${processed.get()} items in $elapsed s. Rate: $rate items/sec")
    }
  }

  val toolOptions = {
    val options = new cli.Options()
    var opt = new cli.Option("i", "input", true, "Documents to predict")
    opt.setRequired(true)
    options.addOption(opt)

    opt = new cli.Option("o", "output", true, "Output CSV file")
    opt.setRequired(true)
    options.addOption(opt)

    opt = new cli.Option("a", "alloy", true, "Alloy file")
    opt.setRequired(true)
    options.addOption(opt)

    opt = new cli.Option("v", "Validate alloy")
    options.addOption(opt)

    opt = new cli.Option("h", "help", false, "Show help screen")
    options.addOption(opt)

    options
  }
}
