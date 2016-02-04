package com.idibon.ml.app

import scala.io.Source
import scala.collection.JavaConverters._
import java.util.concurrent.LinkedBlockingQueue
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods.{compact, render, parse}
import com.idibon.ml.alloy.JarAlloy
import com.idibon.ml.predict._

/** Command-line batch prediction tool
  *
  * Given an alloy and a file containing a list of newline-delimited
  * Idibon JSON documents, predicts all of them and writes the
  * results to an output file
  */
object Predict extends Tool {

  private [this] def parseCommandLine(argv: Array[String]) = {
    val options = (new org.apache.commons.cli.Options)
      .addOption("i", "input", true, "Input file containing documents")
      .addOption("o", "output", true, "Result output CSV file")
      .addOption("a", "alloy", true, "Input alloy")

    (new org.apache.commons.cli.BasicParser).parse(options, argv)
  }

  def run(engine: com.idibon.ml.common.Engine, argv: Array[String]) {
    implicit val formats = org.json4s.DefaultFormats

    val cli = parseCommandLine(argv)

    val model = JarAlloy.load(engine, cli.getOptionValue('a'))
      .asInstanceOf[JarAlloy[Classification]]

    /* results will be written to the output file by a consumer thread;
     * after the last document is predicted, the main thread will post
     * a sentinel value of None to cause the result output thread to
     * terminate. */
    val results = new LinkedBlockingQueue[Option[(JObject, Seq[Classification])]]
    val output = new org.apache.commons.csv.CSVPrinter(
      new java.io.PrintWriter(cli.getOptionValue('o'), "UTF-8"),
      org.apache.commons.csv.CSVFormat.RFC4180)

    val resultsThread = new Thread(new Runnable() {
      override def run {
        var labels: Option[Seq[String]] = None

        Stream.continually(results.take)
          .takeWhile(_.isDefined)
          .foreach(_ match {
            case Some((document, prediction)) => {
              /* initialize the CSV header structure and label output order
               * on the first valid row, after the labels are known */
              if (labels.isEmpty) {
                labels = Some(prediction.map(_.label).sortWith(_ < _))
                output.printRecord((Seq("Name", "Content") ++
                  labels.get.map(l => List(l, s"features[$l]")).flatten).asJava)
              }
              // output the prediction result and original content in JSON
              val labelResults = prediction.sortWith(_.label < _.label).map(r => {
                List(r.probability, r.significantFeatures.map(_._1))
              }).reduce(_ ++ _)
              val row = (Seq(
                (document \ "name").extract[Option[String]].getOrElse(""),
                (document \ "content").extract[String]) ++ labelResults).asJava

              output.printRecord(row)
            }
            case _ => { }
          })
      }
    })
    resultsThread.start

    try {
      Source.fromFile(cli.getOptionValue('i'), "UTF-8")
        .getLines.toStream.par
        .foreach(line => {
          val document = parse(line).extract[JObject]
          val result = model.predict(document,
            (new PredictOptionsBuilder).showSignificantFeatures(0.1f).build)
          results.offer(Some((document, result.asScala)))
        })
    } finally {
      // send the sentinel value to shut down the output thread
      results.offer(None)
      // wait for the output thread to finish and close the stream
      resultsThread.join
      output.close
    }
  }
}
