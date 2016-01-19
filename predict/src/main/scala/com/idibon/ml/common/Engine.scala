package com.idibon.ml.common

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Engine Traits.
  */

trait Engine {

  val sparkContext: SparkContext

}

abstract class BaseEngine extends Engine {

  // Instantiate the Spark environment
  val sparkContext = BaseEngine.sparkContext

}

/**
  * Currently only one SparkContext can exist per JVM, hence the use of this companion object
  */
object BaseEngine {
  val sparkContext = {
    val conf = new SparkConf().setAppName("idiml").setMaster("local[3]").set("spark.driver.host", "localhost")
    new SparkContext(conf)
  }
}

trait PredictEngine extends Engine {

  def start(modelPath: String, documentPath: String)

}

trait TrainEngine extends Engine {

  def start(infilePath: String, modelPath: String)

}
