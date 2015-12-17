import scala.collection.immutable.StringLike
import scala.reflect.runtime.universe._

import com.idibon.ml.feature.{Feature,FeatureTransformer}

package com.idibon.ml.feature.tokenizer {

  /** Internal implementation of the tokenization transformation */
  private[tokenizer] class Tokenizer {

    /** Partitions an input content string into the tokens it contains */
    def tokenize(content: StringLike[_]): Seq[Token] = {
      content.split(' ').map(t => new Token(t, Tag.Word, 0, 0))
    }
  }

  /** Tokenization FeatureTransformer */
  class TokenTransformer extends Tokenizer with FeatureTransformer {

    /** Tokenizes an array of strings stored in the "content" key of the map
      *
      * If more than one input string is provided, the result will be the
      * concatenation of the tokenized results of all input strings.
      */
    def apply(contents: Seq[Feature[String]]): Seq[Token] = {
      contents.foldLeft(List[Token]())((t,c) => t ++ tokenize(c.get))
    }
  }
}
