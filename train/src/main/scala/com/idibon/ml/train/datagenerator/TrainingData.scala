package com.idibon.ml.train.datagenerator

import org.json4s.JObject

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

    /** True if the position is within the span */
    def inside(x: Int): Boolean = {
      offset.map(offs => x >= offs && x < end.get).getOrElse(true)
    }
  }

  /** JSON schema for the annotation's label hash
    *
    * @param name label name
    */
  case class LabelName(name: String)
}
