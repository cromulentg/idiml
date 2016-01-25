package com.idibon.ml.feature.wordshapes

import java.io.{DataOutputStream, DataInputStream}

import com.idibon.ml.alloy.Codec
import com.idibon.ml.feature.{Feature, Buildable, Builder}

/**
  * This class represents a word shape.
  */
case class Shape(shape: String) extends Feature[Shape]
  with Buildable[Shape, ShapeBuilder] {

  def get = this

  /** Stores the feature to an output stream so it may be reloaded later. */
  override def save(output: DataOutputStream): Unit = {
    Codec.String.write(output, shape)
  }
}

class ShapeBuilder extends Builder[Shape] {
  /** Reloads a previously saved feature */
  override def build(input: DataInputStream): Shape = {
    val shape = Codec.String.read(input)
    new Shape(shape)
  }
}
