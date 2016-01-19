package com.idibon.ml.predict

import com.idibon.ml.common.Engine
import org.apache.spark.{SparkContext, SparkConf}
import com.idibon.ml.alloy.ScalaJarAlloy
import com.typesafe.scalalogging.StrictLogging
import org.json4s._
import org.json4s.JsonDSL._

/**
  * Toy engine class that stitches together Idibon's feature pipeline and Spark's LR
  * to perform a prediction.
  */
class EmbeddedEngine extends Engine with StrictLogging{

  val sparkContext = EmbeddedEngine.sparkContext

  /**
    * Very crude POC. This will change as we add more to the code base.
    */
  override def start(modelPath: String) = {
    val model = ScalaJarAlloy.load(this, modelPath)

    val text: String = "Everybody loves replacing hadoop with spark because it's much faster. a b d"
//    val text: String = "I am very neutral at the moment"
    val doc: JObject = ( "content" -> text )

    val result = model.predict(doc, new PredictOptionsBuilder().build())
    logger.info(result.toString)
  }

  override def start(infilePath: String, modelPath: String): Unit = {}
}

/**
  * Currently only one SparkContext can exist per JVM, hence the use of this companion object
  */
object EmbeddedEngine {
  val sparkContext = {
    val conf = new SparkConf().setMaster("local").setAppName("idiml")
    new SparkContext(conf)
  }
}
