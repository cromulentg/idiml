package com.idibon.ml.test

import com.idibon.ml.common.EmbeddedEngine

/**
  * Provides a SparkContext for testing.
  */

object Spark {
  private [this] val _engine = new EmbeddedEngine
  val sc = _engine.sparkContext
}
