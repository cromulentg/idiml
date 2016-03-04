package com.idibon.ml.feature.wordshapes

import com.idibon.ml.feature.{Chain, Feature, FeatureTransformer}
import com.idibon.ml.feature.tokenizer.Token

/** WordShape FeatureTransformer
  *
  * Produces a bag of wordshapes, much like bag of words
  * with an extra transformation.
  * Preserves the order of token sequence.
  *
  * */
class WordShapesTransformer extends FeatureTransformer with ToShape {

  /** Produces a sequence of shapes from a sequence of Tokens.
    *
    * @param tokens
    * @return wordshapes represented as a sequence of Shape features.
    */
  def apply(tokens: Seq[Feature[Token]]): Seq[Shape] = {
    //return the sequence of tokens translated to shapes (in the same order)
    tokens.map(token => new Shape(toShape(token.get.content)))
  }
}

/** WordShapesTransformer for sequence classifiers */
class ChainWordShapesTransformer extends FeatureTransformer with ToShape {
  def apply(tokens: Chain[Feature[Token]]): Chain[Shape] = {
    tokens.map(link => Shape(toShape(link.value.get.content)))
  }
}

/** Mixin trait for generating word "shapes" */
trait ToShape {
  /** Produces a word shape representation string from a token
    *
    * @param token
    * @return shape represented as a string.
    */
  def toShape(token: String): String = {
    val shape = token.replaceAll("\\p{Lu}", "C")
      .replaceAll("CC+", "CC")
      .replaceAll("\\p{Ll}", "c")
      .replaceAll("cc+", "cc")
      .replaceAll("\\p{Nd}", "n")
      .replaceAll("nn+", "nn")
      .replaceAll("\\p{Zs}+", "s")
      .replaceAll("[^snCc]+", "p")

    return shape
  }

}
