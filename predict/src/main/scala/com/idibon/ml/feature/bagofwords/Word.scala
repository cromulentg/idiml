package com.idibon.ml.feature.bagofwords

import java.io.{DataOutputStream, DataInputStream}

import com.idibon.ml.alloy.Codec
import com.idibon.ml.feature.Feature

/** This class represents a word and it's occurrence count.
  * @author Stefan Krawczyk <stefan@idibon.com>
  */
case class Word(var word: String, var count: Int) extends Feature[Word] {

  def get = this

  /** Stores the feature to an output stream so it may be reloaded later. */
  override def save(output: DataOutputStream): Unit = {
    Codec.String.write(output, word)
    Codec.VLuint.write(output, count)
  }

  /** Reloads a previously saved feature */
  override def load(input: DataInputStream): Unit = {
    word = Codec.String.read(input)
    count = Codec.VLuint.read(input)
  }
}
