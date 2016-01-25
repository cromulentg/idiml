package com.idibon.ml.feature.bagofwords

import com.idibon.ml.alloy.Codec
import com.idibon.ml.feature._

/** This class represents a word and it's occurrence count.
  * @author Stefan Krawczyk <stefan@idibon.com>
  */
case class Word(word: String, count: Int) extends Feature[Word]
    with Buildable[Word, WordBuilder] {

  def get = this

  /** Stores the feature to an output stream so it may be reloaded later. */
  override def save(output: FeatureOutputStream): Unit = {
    Codec.String.write(output, word)
    Codec.VLuint.write(output, count)
  }
}

class WordBuilder extends Builder[Word] {
  /** Reloads a previously saved feature */
  override def build(input: FeatureInputStream): Word = {
    val word = Codec.String.read(input)
    val count = Codec.VLuint.read(input)
    new Word(word, count)
  }
}
