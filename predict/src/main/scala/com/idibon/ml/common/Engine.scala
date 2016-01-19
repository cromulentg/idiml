package com.idibon.ml.common

import org.apache.spark.SparkContext

trait Engine {

  val sparkContext: SparkContext

}
