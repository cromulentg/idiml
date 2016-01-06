package com.idibon.ml.feature

import org.json4s._

/** Extracts document content from a JObject and turns it into a list of StringFeatures */
class DocumentExtractor extends FeatureTransformer {
  def apply(document: JObject): Seq[StringFeature] = {
    List(StringFeature((document \ "content").asInstanceOf[JString].s))
  }
}
