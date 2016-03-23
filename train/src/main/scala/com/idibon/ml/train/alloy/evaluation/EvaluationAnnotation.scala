package com.idibon.ml.train.alloy.evaluation

import com.idibon.ml.feature.tokenizer.Token
import com.idibon.ml.predict.crf.BIOType

/**
  * Class for storing annotations good for evaluation purposes.
  *
  * Span annotations must include both offset and length
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/21/16.
  *
  * @param label annotation label
  * @param isPositive true for positive training items
  * @param offset start of the span (span annotations)
  * @param length length of the span (span annotations)
  * @param tokens tokens that make up the span.
  * @param tokenTags token tags that make up the tokens provided.
  */
case class EvaluationAnnotation(label: LabelName, isPositive: Boolean,
                                offset: Option[Int], length: Option[Int],
                                tokens: Option[Seq[Token]] = None,
                                tokenTags: Option[Seq[BIOType.Value]] = None) {

  def this(label: String, isPositive: Boolean,
           offset: Option[Int], length: Option[Int],
           tokens: Option[Seq[Token]],
           tokenTags: Option[Seq[BIOType.Value]]) =
    this(LabelName(label), isPositive, offset, length, tokens, tokenTags)

  /** True if the annotation is a valid span annotation */
  def isSpan = offset.isDefined && length.isDefined

  /** End position of this span, exclusive */
  val end: Option[Int] = offset.map(_ + length.get)

  def contains(t: Token): Boolean = {
    offset.map(o => { t.end > o  && t.offset < end.get }).getOrElse(true)
  }

  /** True if the position is within the span */
  def inside(x: Int): Boolean = {
    offset.map(offs => x >= offs && x < end.get).getOrElse(true)
  }

  /** Helper method to return tokens matched up with their tags **/
  def getTokensNTags(): Seq[(Token, BIOType.Value)] = {
    tokens.getOrElse(Seq()).zip(tokenTags.getOrElse(Seq()))
  }
}

case class LabelName(name: String)
