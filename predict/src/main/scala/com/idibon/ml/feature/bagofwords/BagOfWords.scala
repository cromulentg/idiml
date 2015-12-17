package com.idibon.ml.feature.bagofwords

import com.idibon.ml.feature.{Feature,FeatureTransformer}
import com.idibon.ml.feature.tokenizer.Token
import scala.reflect.runtime.universe.typeOf

/** BagOfWords FeatureTransformer
  *
  * Produces a bag of words. i.e. a Word Feature for each unique word that
  * doesn't care about the order at all.
  *
  * The word feature also stores the count of occurrences in case someone
  * wants to use that downstream.
  * */
class BagOfWordsTransformer extends FeatureTransformer {

  /** Produces a bag of words from a sequence of Tokens.
    *
    * @param inputFeatures
    * @return bag of words represented as a sequence of Word features.
    */
  def apply(tokens: Seq[Feature[Token]]): Seq[Word] = {
    // then, iterating over the sequence, accumulate a map of word -> count
    val wordMap: Map[String, Int] = tokens.foldLeft(Map.empty[String, Int])(
      (mp, token) => mp + (token.get.content -> (mp.getOrElse(token.get.content, 0) + 1)))
    // convert map into a sequence of WordFeatures & return
    wordMap.map{ case (k, v) => new Word(k, v) }.toSeq
  }
}
