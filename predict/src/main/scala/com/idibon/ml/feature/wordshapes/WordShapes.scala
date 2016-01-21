package com.idibon.ml.feature.wordshapes

//TODO: unclear if there should be a separate 'shape' class or if word is the appropriate type here
import com.idibon.ml.feature.bagofwords.Word

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
    * @return wordshapes represented as a sequence of Word features.
    */
  def apply(tokens: Seq[Feature[Token]]): Seq[Word] = {
    //iterate over the sequence, converting each token to its shape and updated counts in the map
    val shapeMap: Map[String, Int] = tokens.foldLeft(Map.empty[String, Int])(
      (mp, token) => mp + (toShape(token.get.content) -> (mp.getOrElse(toShape(token.get.content), 0) + 1)))

    // convert map into a sequence of WordFeatures & return
    shapeMap.map{ case (k, v) => new Word(k, v) }.toSeq
  }

  //this is the shape/structure ididat uses,
  //is there a more scala way to do this
  //and is this still correct?
  /** Produces a wordshape representation from a string
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
