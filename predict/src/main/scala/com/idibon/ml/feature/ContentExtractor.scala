package com.idibon.ml.feature

import org.json4s._

/** Extracts document content from a JObject and turns it as a StringFeature */
class ContentExtractor extends FeatureTransformer {
  def apply(document: JObject): StringFeature = {
    StringFeature((document \ "content").asInstanceOf[JString].s)
  }
}
