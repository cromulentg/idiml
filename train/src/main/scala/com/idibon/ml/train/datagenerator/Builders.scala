package com.idibon.ml.train.datagenerator

import org.json4s.ShortTypeHints
import org.json4s.native.Serialization
import org.json4s.native.Serialization.{writePretty}

/**
  * Static object to house global defaults for data generator builders.
  */
object BuilderDefaults {
  implicit val formats = Serialization.formats(ShortTypeHints(List(
    classOf[MultiClassDataFrameGeneratorBuilder],
    classOf[KClassDataFrameGeneratorBuilder]
  )))
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

class MultiClassDataFrameGeneratorBuilder() extends SparkDataGeneratorBuilder {
  override def build(): MultiClassDataFrameGenerator = new MultiClassDataFrameGenerator()
}

class KClassDataFrameGeneratorBuilder() extends SparkDataGeneratorBuilder {
  override def build(): KClassDataFrameGenerator = new KClassDataFrameGenerator()
}
