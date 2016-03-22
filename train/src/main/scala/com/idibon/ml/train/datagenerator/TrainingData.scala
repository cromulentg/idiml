package com.idibon.ml.train.datagenerator

import com.idibon.ml.feature.tokenizer.Token

/** Defines various document training data types */
package json {

  /** The JSON schema for documents processed by the data generators
    *
    * Documents may include additional fields (e.g., metadata) used by
    * feature extraction; however, every document used for training must
    * include content and annotations as a minimum
    *
    * @param content the document content
    * @param annotations annotations list
    */
  case class Document(content: String, annotations: List[Annotation])

  /** JSON schema for annotations attached to training documents
    *
    * Span annotations must include both offset and length
    *
    * @param label annotation label
    * @param isPositive true for positive training items
    * @param offset start of the span (span annotations)
    * @param length length of the span (span annotations)
    */
  case class Annotation(label: LabelName, isPositive: Boolean,
    offset: Option[Int], length: Option[Int]) {
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
  }

  object Annotation {
    def apply(l: String, b: Boolean, o: Int, i: Int): Annotation =
      this(LabelName(l), b, Some(o), Some(i))
    def apply(l: String, b: Boolean): Annotation =
      this(LabelName(l), b, None, None)
    def apply(l: String): Annotation =
      this(LabelName(l), false, None, None)
  }

  /** JSON schema for the annotation's label hash
    *
    * @param name label name
    */
  case class LabelName(name: String)
}
