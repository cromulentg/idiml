package com.idibon.ml.common

import java.security.MessageDigest
import java.io.File

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
private [common] object EmbeddedEngine extends StrictLogging {
  val sparkContext = {
    /* limit the number of Spark workers to the lesser of:
     * - the number of virtual CPUs available, OR
     * - the number of available 200M heap allocations possible
     * this is primarily for CircleCI, which exposes 32 virtual CPU
     * cores but is limited to ~1.5GB of heap. */
    val cpuCores = Runtime.getRuntime().availableProcessors()
    val maxHeap = Runtime.getRuntime().maxMemory() / 1048576
    val workers = Math.max(2, Math.min(cpuCores, maxHeap / 200))

    logger.info(s"Using $workers Spark workers (CORES=$cpuCores HEAP=$maxHeap)")

    val conf = new SparkConf().setAppName("idiml")
      .set("spark.driver.host", "localhost")
      .set("spark.ui.enabled", "false")
      .setMaster(s"local[$workers]")
    new SparkContext(conf)
  }

  /** Encodes a username into a unique filesystem-safe path
    *
    * Uses the SHA1 hashing algorithm to convert a username into 12
    * hexadecimal characters
    */
  def encodeUsername(name: String): String =
    MessageDigest.getInstance("SHA1")
      .digest(name.getBytes("UTF-8"))
      .take(6)
      .map(v => f"$v%02x")
      .mkString("")

  /** Private, per-user temporary directory for IdiML use
    *
    * Replaces java.io.tmpdir with a per-user subdirectory so that access
    * denied errors aren't generated on multi-user systems if one user
    * creates a global IdiML temp directory with o+r rather than o+rw.
    */
  val idimlTemp = {
    val systemTemp = System.getProperty("java.io.tmpdir")
    val perUserTemp = System.getProperty("user.name") match {
      case x: String if !x.isEmpty => {
        // create a directory for the current user
        new File(systemTemp, s"idiml-${encodeUsername(x)}")
      }
      case _ => {
        // if there is no current user, just use the system temp
        new File(systemTemp, "idiml")
      }
    }
    logger.info(s"Setting temporary directory to $perUserTemp")
    perUserTemp.mkdirs()
    System.setProperty("java.io.tmpdir", perUserTemp.getAbsolutePath())
  }
}
