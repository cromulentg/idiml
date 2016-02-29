package com.idibon.ml.feature.wordshapes

import com.idibon.ml.alloy.Codec
import com.idibon.ml.feature._

/**
  * This class represents a word shape.
  */
case class Shape(shape: String) extends Feature[Shape]
  with Buildable[Shape, ShapeBuilder] {

  def get = this

  /** Stores the feature to an output stream so it may be reloaded later. */
  override def save(output: FeatureOutputStream): Unit = {
    Codec.String.write(output, shape)
  }

  def getHumanReadableString: Option[String] = None
}

class ShapeBuilder extends Builder[Shape] {
  /** Reloads a previously saved feature */
  override def build(input: FeatureInputStream): Shape = {
    val shape = Codec.String.read(input)
    new Shape(shape)
  }
}
