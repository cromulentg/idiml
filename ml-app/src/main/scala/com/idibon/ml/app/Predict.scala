package com.idibon.ml.app

import scala.io.Source
import scala.collection.JavaConverters._
import java.util.concurrent.LinkedBlockingQueue
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods.{compact, render, parse}
import com.idibon.ml.alloy.ScalaJarAlloy
import com.idibon.ml.predict.{PredictModel, PredictResult, PredictOptions}

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
      .addOption("o", "output", true, "Result output file")
      .addOption("a", "alloy", true, "Input alloy")

    (new org.apache.commons.cli.BasicParser).parse(options, argv)
  }

  def run(engine: com.idibon.ml.common.Engine, argv: Array[String]) {
    implicit val formats = org.json4s.DefaultFormats

    val cli = parseCommandLine(argv)

    val model = ScalaJarAlloy.load(engine, cli.getOptionValue('a'))

    /* results will be written to the output file by a consumer thread;
     * after the last document is predicted, the main thread will post
     * a sentinel value of None to cause the result output thread to
     * terminate. */
    val results = new LinkedBlockingQueue[Option[(JObject, Map[String, PredictResult])]]
    val output = new java.io.PrintWriter(cli.getOptionValue('o'), "UTF-8")

    val resultsThread = new Thread(new Runnable() {
      override def run {
        Stream.continually(results.take)
          .takeWhile(_.isDefined)
          .foreach(_ match {
            case Some((document, prediction)) => {
              // output the prediction result and original content in JSON
              val line = compact(render(
                ("content" -> (document \ "content").extract[String]) ~
                  ("prediction" -> prediction.toString)))
              output.println(line)
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
          val result = model.predict(document, PredictOptions.DEFAULT)
          results.offer(Some((document, result.asScala.toMap)))
        })
    } finally {
      results.offer(None)
    }
  }
}
