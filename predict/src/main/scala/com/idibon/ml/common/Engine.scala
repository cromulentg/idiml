package com.idibon.ml.common

import org.apache.spark.SparkContext

trait Engine {

  val sparkContext: SparkContext

  def start(modelPath: String)

  def start(infilePath: String, modelPath: String)
}
