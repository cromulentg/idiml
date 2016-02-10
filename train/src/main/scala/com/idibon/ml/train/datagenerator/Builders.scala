package com.idibon.ml.train.datagenerator

import com.idibon.ml.train.datagenerator.scales._
import org.json4s.ShortTypeHints
import org.json4s.native.Serialization
import org.json4s.native.Serialization.{writePretty}

/**
  * Static object to house global defaults for data generator builders.
  */
object BuilderDefaults {
  val classHints = List(
    classOf[MultiClassDataFrameGeneratorBuilder],
    classOf[KClassDataFrameGeneratorBuilder])
  implicit val formats = Serialization.formats(ShortTypeHints(classHints))
}

trait SparkDataGeneratorBuilder {

  def build(): SparkDataGenerator

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

case class MultiClassDataFrameGeneratorBuilder(var scale: DataSetScaleBuilder = new NoOpScaleBuilder())
  extends SparkDataGeneratorBuilder {
  override def build(): MultiClassDataFrameGenerator = {
    new MultiClassDataFrameGenerator(this)
  }
}

case class KClassDataFrameGeneratorBuilder(var scale: DataSetScaleBuilder = new BalancedBinaryScaleBuilder())
  extends SparkDataGeneratorBuilder {

  override def build(): KClassDataFrameGenerator = new KClassDataFrameGenerator(this)
}
