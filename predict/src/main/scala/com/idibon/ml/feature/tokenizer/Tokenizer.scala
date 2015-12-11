import com.idibon.ml.feature.{Feature,FeatureTransformer}
import scala.collection.immutable.StringLike
import scala.reflect.runtime.universe._

package com.idibon.ml.feature.tokenizer {

  /** Internal implementation of the tokenization transformation */
  private[tokenizer] class Tokenizer {

    /** Partitions an input content string into the tokens it contains */
    def tokenize(content: StringLike[_]): Seq[Token] = {
      content.split(' ').map(t => new Token(t, Tag.Word, 0, 0))
    }
  }

  /** Tokenization FeatureTransformer */
  class TokenTransformer extends Tokenizer with FeatureTransformer[Token] {

    def input = TokenTransformer.input

    def options = None

    /** Tokenizes an array of strings stored in the "content" key of the map
      * 
      * If more than one input string is provided, the result will be the
      * concatenation of the tokenized results of all input strings.
      */
    def apply(inputFeatures: Map[String, Seq[Feature[_]]]): Seq[Token] = {
      inputFeatures("content")
        .map(f => tokenize(f.getAs[String]))
        .flatten
    }
  }

  private[tokenizer] object TokenTransformer {
    lazy val input = Map("content" -> typeOf[String])
  }
}
