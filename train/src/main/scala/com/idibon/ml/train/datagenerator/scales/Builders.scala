package com.idibon.ml.train.datagenerator.scales

import org.json4s.ShortTypeHints
import org.json4s.native.Serialization
import org.json4s.native.Serialization._

/**
  * Static object to house global defaults for data generator builders.
  */
object BuilderDefaults {

  val DEFAULT_TOLERANCE = 0.66

  val classHints = List(
    classOf[NoOpScaleBuilder],
    classOf[BalancedBinaryScaleBuilder])
  implicit val formats = Serialization.formats(ShortTypeHints(classHints))
}

trait DataSetScaleBuilder {

  def build(): DataSetScale

  /**
    * Creates a pretty printed JSON string.
    *
    * This will be useful for building a tool to output some nice JSON configuration.
    *
    * @return
    */
  override def toString(): String = {
    implicit val formats = BuilderDefaults.formats
    writePretty(this)
  }
}

case class NoOpScaleBuilder() extends DataSetScaleBuilder {
  override def build(): NoOpScale = new NoOpScale()
}

case class BalancedBinaryScaleBuilder(private[scales] var tolerance: Double = BuilderDefaults.DEFAULT_TOLERANCE,
                                      private[scales] var seed: Long = System.currentTimeMillis())
  extends DataSetScaleBuilder {
  override def build(): BalancedBinaryScale = new BalancedBinaryScale(this)
}
