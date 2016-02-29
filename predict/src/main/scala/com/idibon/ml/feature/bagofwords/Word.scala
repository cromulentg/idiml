package com.idibon.ml.feature.bagofwords

import com.idibon.ml.alloy.Codec
import com.idibon.ml.feature._

/** This class represents a word
  * @author Stefan Krawczyk <stefan@idibon.com>
  */
case class Word(word: String) extends Feature[Word]
    with Buildable[Word, WordBuilder] {

  def get = this

  /** Stores the feature to an output stream so it may be reloaded later. */
  override def save(output: FeatureOutputStream): Unit = {
    Codec.String.write(output, word)
  }

  def getAsString: Option[String] = Some(this.word)
}

class WordBuilder extends Builder[Word] {
  /** Reloads a previously saved feature */
  override def build(input: FeatureInputStream): Word = {
    val word = Codec.String.read(input)
    new Word(word)
  }
}
