package com.idibon.ml.common

/** The Engine is the primary unit of context within idiml
  *
  * It provides access to system-level integration, such as the active
  * SparkContext.
  */
trait Engine {

  /** Returns the Spark context available to the engine. */
  val sparkContext: org.apache.spark.SparkContext
}
