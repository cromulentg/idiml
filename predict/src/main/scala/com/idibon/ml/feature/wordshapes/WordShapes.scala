package com.idibon.ml.feature.wordshapes

import com.idibon.ml.feature.{Feature,FeatureTransformer}
import com.idibon.ml.feature.tokenizer.Token
import scala.reflect.runtime.universe.typeOf

/** WordShape FeatureTransformer
  *
  * Produces a bag of wordshapes, much like bag of words with an extra transformation,
  * doesn't care about the order at all.
  *
  * Includes count of occurrences of each shape
  * */
class WordShapesTransformer extends FeatureTransformer {

  /** Produces wordshapes from a sequence of Tokens.
    *
    * @param tokens
    * @return wordshapes represented as a sequence of Shape features.
    */
  def apply(tokens: Seq[Feature[Token]]): Seq[Shape] = {
    //iterates over the sequence, converting each token to its shape and updating counts in the map
    val shapeMap: Map[String, Int] = tokens.foldLeft(Map.empty[String, Int])(
      (mp, token) => mp + (toShape(token.get.content) -> (mp.getOrElse(toShape(token.get.content), 0) + 1)))

    // convert map into a sequence of ShapeFeatures & return
    shapeMap.map{ case (k, v) => new Shape(k, v) }.toSeq
  }

  /** Produces a word shape representation string from a token
    *
    * @param token
    * @return shape represented as a string.
    */
  def toShape(token: String): String = {
    val shape = token.replaceAll("[A-Z]", "C")
      .replaceAll("CC+", "CC")
      .replaceAll("[a-z]", "c")
      .replaceAll("cc+", "cc")
      .replaceAll("[0-9]", "n")
      .replaceAll("nn+", "nn")
      .replaceAll("\\s+", "s")
      .replaceAll("[^snCc]+", "p")

    return shape
  }
}
