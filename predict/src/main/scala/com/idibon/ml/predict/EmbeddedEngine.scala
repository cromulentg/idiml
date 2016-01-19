package com.idibon.ml.predict

import java.io.File
import com.idibon.ml.common.{BaseEngine, PredictEngine}
import org.apache.spark.{SparkContext, SparkConf}
import com.idibon.ml.alloy.ScalaJarAlloy
import com.typesafe.scalalogging.StrictLogging
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods._
import org.json4s.native.JsonMethods.{parse, render, compact}

import scala.io.Source

/**
  * Toy engine class that stitches together Idibon's feature pipeline and Spark's LR
  * to perform a prediction.
  */
class EmbeddedEngine extends BaseEngine with PredictEngine with StrictLogging{

  /**
    * Very crude POC. This will change as we add more to the code base.
    *
    * @param modelPath the path to load the model from.
    * @param documentPath the path to load the new-line separated JSON documents for classification.
    */
  override def start(modelPath: String, documentPath: String) = {
    implicit val formats = org.json4s.DefaultFormats
    val model = ScalaJarAlloy.load(this, modelPath)
    val results = Source.fromFile(new File(documentPath)).getLines().toStream.par.foreach(line => {
      val doc = parse(line).extract[JObject]
      val result = model.predict(doc, new PredictOptionsBuilder().build())
      logger.info(s"input=[${(doc \ "content").extract[String]}]\noutput=[${result.toString}")
    })
  }

}
