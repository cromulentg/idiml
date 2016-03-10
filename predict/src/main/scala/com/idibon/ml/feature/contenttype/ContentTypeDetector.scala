package com.idibon.ml.feature.contenttype

import com.idibon.ml.feature.FeatureTransformer
import org.json4s.{JString, JObject}

/** Generates a feature based on the document's detected content-type.
  *
  * This can be used to guide markup-aware tokenization algorithms.
  */
class ContentTypeDetector extends FeatureTransformer {

  /** Uses document starting tags or metadata fields to detect
    * document's content type. Defaults to plaintext
    *
    * @param document a parsed JSON document
    * @return a ContentType feature
    */
  def apply(document: JObject): ContentType = {
    (document \ "content").toOption //first check starting tags
      .flatMap(content => startsWithTag(content.asInstanceOf[JString].s))
      .orElse({
        (document \ "metadata").toOption //second, check metadata fields
          .flatMap(metadata => metadataRules(metadata.asInstanceOf[JObject]))
      }) //default to plain text
      .getOrElse(ContentType(ContentTypeCode.PlainText))
  }

  /** Uses starts with test to check for known starting tags of types
    *
    * Returns None if no check passes
    */
  private def startsWithTag(content: String): Option[ContentType] = {
    val htmlTag = "<!doctype "
    val xmlTag = "<?xml "
    Some(content)
      .filter(_.regionMatches(true, 0, htmlTag, 0, htmlTag.length()))
      .flatMap(o => Some(ContentType(ContentTypeCode.HTML)))
      .orElse({
        Some(content)
          .filter(_.regionMatches(true, 0, xmlTag, 0, xmlTag.length()))
          .flatMap(o => Some(ContentType(ContentTypeCode.XML)))
      })
  }

  /** Hardcoded checks for known metadata fields that imply
    * a specific document type
    *
    * Returns None if no check passes
    */
  private def metadataRules(metadata: JObject): Option[ContentType] = {
    (metadata \ "lexisnexis").toOption
      .flatMap(metadata => Some(ContentType(ContentTypeCode.HTML)))
      .orElse({
        (metadata \ "newscred").toOption
          .flatMap(metadata => Some(ContentType(ContentTypeCode.HTML)))
      })
  }
}

