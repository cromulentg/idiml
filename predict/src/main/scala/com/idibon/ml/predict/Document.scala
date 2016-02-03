package com.idibon.ml.predict

import org.apache.spark.mllib.linalg.Vector
import org.json4s._

/** Documents are the base unit for predictive analytics, and cache various
  * state as the analysis proceeds through the Alloy.
  *
  * This enables models to choose the best representation of the content for
  * the operations
  *
  * @param json - The original document JSON object
  * @param featureVector - the document transformed by a feature pipeline
  * @param featureTable - a mapping of 1-hot feature encodings (dimensions in
  *    featureVector) to human-readable representations
  */
case class Document(json: JObject,
  featureVector: Option[Vector],
  featureTable: Option[Map[Int, String]])

object Document {
  /** Builder method for documents with only JSON representations */
  def document(doc: JObject) = new Document(doc, None, None)
}
