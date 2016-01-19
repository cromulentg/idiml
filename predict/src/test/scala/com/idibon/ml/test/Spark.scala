package com.idibon.ml.test

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Provides a SparkContext for testing.
  */

object Spark {

  private val conf = new SparkConf()
    .setMaster("local[2]")
    .setAppName("idiML")
  lazy val sc: SparkContext = new SparkContext(conf)

}
