package com.idibon.ml.predict

import com.idibon.ml.feature.Feature
import org.apache.spark.mllib.linalg.Vector
import org.json4s._

/** Documents are the base unit for predictive analytics, and cache various
  * state as the analysis proceeds through the Alloy.
  *
  * This enables models to choose the best representation of the content for
  * the operations
  *
  * @param json the original document JSON object
  * @param transformed the document transformed by a feature pipeline, and
  *   a function that inverts the mapping for an aribtrary vector
  */
case class Document(json: JObject,
  transformed: Option[(Vector, (Vector) => Seq[Option[Feature[_]]])])

object Document {
  /** Builder method for documents with only JSON representations */
  def document(doc: JObject) = new Document(doc, None)
}
