package com.idibon.ml.feature.contenttype

import com.idibon.ml.feature.FeatureTransformer
import org.json4s.JObject

/** Generates a feature based on the document's detected content-type.
  *
  * This can be used to guide markup-aware tokenization algorithms.
  */
class ContentTypeDetector extends FeatureTransformer {

  /** Uses document metadata or magic-byte detection to determine a
    * document's content type.
    *
    * @param document a parsed JSON document
    * @return a ContentType feature
    */
  def apply(document: JObject): ContentType = {
    // TODO: implement!
    ContentType(ContentTypeCode.PlainText)
  }
}
