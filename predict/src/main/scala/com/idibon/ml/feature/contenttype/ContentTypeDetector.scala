package com.idibon.ml.feature.contenttype

import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.feature.FeatureTransformer
import com.idibon.ml.alloy.Alloy

import org.json4s.{JString, JObject}
import org.json4s.JsonDSL._

/** Generates a feature based on the document's detected content-type.
  *
  * This can be used to guide markup-aware tokenization algorithms.
  *
  * @param deepDetection sniffs the entire document, rather than just
  *    the first line, to try to detect the document type
  */
class ContentTypeDetector(deepDetection: Boolean) extends FeatureTransformer
    with Archivable[ContentTypeDetector, ContentTypeDetectorLoader] {

  /** Uses document starting tags or metadata fields to detect
    * document's content type. Defaults to plaintext
    *
    * @param document a parsed JSON document
    * @return a ContentType feature
    */
  def apply(document: JObject): ContentType = {
    val content = (document \ "content").toOption

    content
      // first check starting tags
      .flatMap(text => startsWithTag(text.asInstanceOf[JString].s))
      // then use metadata, if present
      .orElse({
        (document \ "metadata").toOption
          .flatMap(metadata => metadataRules(metadata.asInstanceOf[JObject]))
      })
      // finally, fallback to sniffing the entire text
      .orElse({
        content.flatMap(text => sniffContent(text.asInstanceOf[JString].s))
      })
      .getOrElse(ContentType(ContentTypeCode.PlainText))
  }

  def save(writer: Alloy.Writer): Option[JObject] = {
    Some(("deepDetection" -> deepDetection))
  }

  /** "Sniffs" the document content to determine the content type
    *
    * Searches for text snippets with known tags (such as line and
    * paragraph breaks) that are strongly suggestive of HTML markup
    *
    * @param text content to analyze
    * @return the content type if detected, else None
    */
  private def sniffContent(content: String): Option[ContentType] = {
    if (deepDetection) {
      None
    } else {
      None
    }
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

/** Paired loader class for content type detector */
class ContentTypeDetectorLoader extends ArchiveLoader[ContentTypeDetector] {

  /** Loads a content type detector from an alloy or config file */
  def load(e: Engine, r: Option[Alloy.Reader], config: Option[JObject]) = {
    implicit val formats = org.json4s.DefaultFormats
    val contentTypeConfig = config.map(_.extract[ContentTypeConfig])
      .getOrElse(ContentTypeConfig(false))
    new ContentTypeDetector(contentTypeConfig.deepDetection)
  }
}

/** JSON configuration schema for content-type detector transform
  *
  * @param deepDetection enable deep content detection
  */
sealed case class ContentTypeConfig(deepDetection: Boolean)
