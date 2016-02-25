package com.idibon.ml.common

import org.apache.spark.{SparkConf, SparkContext}
import com.typesafe.scalalogging.StrictLogging

/** Engine implementation using an in-process, embedded SparkContext */
class EmbeddedEngine extends Engine {

  /** Returns an embedded SparkContext */
  val sparkContext = EmbeddedEngine.sparkContext
}

/** Global companion object to adhere to the 1-SparkContext-per-JVM
  * restriction imposed by Spark
  */
private [this] object EmbeddedEngine {
  val sparkContext = {
    /* limit the number of Spark workers to the lesser of:
     * - the number of virtual CPUs available, OR
     * - the number of available 200M heap allocations possible
     * this is primarily for CircleCI, which exposes 32 virtual CPU
     * cores but is limited to ~1.5GB of heap. */
    val cpuCores = Runtime.getRuntime().availableProcessors()
    val maxHeap = Runtime.getRuntime().maxMemory() / 1048576
    val workers = Math.max(2, Math.min(cpuCores, maxHeap / 200))

    val conf = new SparkConf().setAppName("idiml")
      .set("spark.driver.host", "localhost")
      .set("spark.ui.enabled", "false")
      .setMaster(s"local[$workers]")
    new SparkContext(conf)
  }
}
