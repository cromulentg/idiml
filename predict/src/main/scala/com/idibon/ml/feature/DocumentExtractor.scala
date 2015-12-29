package com.idibon.ml.feature

import org.json4s._

/**
  * Created by michelle on 12/28/15.
  */
class DocumentExtractor extends FeatureTransformer {
  def apply(document: JObject): Seq[StringFeature] = {
    List(StringFeature((document \ "content").asInstanceOf[JString].s))
  }
}
