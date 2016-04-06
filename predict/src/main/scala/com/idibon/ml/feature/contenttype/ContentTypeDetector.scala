package com.idibon.ml.feature.contenttype

import scala.util.Try

import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.feature.tokenizer.XMLBreakIterator
import com.idibon.ml.feature.FeatureTransformer
import com.idibon.ml.alloy.Alloy

import org.json4s.{JString, JObject}
import org.json4s.JsonDSL._

import com.ibm.icu.text.{UCharacterIterator, UForwardCharacterIterator}

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
        content.flatMap(text => {
          if(deepDetection)
            Try(sniffContent(text.asInstanceOf[JString].s)).getOrElse(None)
          else
            None
        })
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
  private def sniffContent(text: String): Option[ContentType] = {
    var offs = text.indexOf('<')
    var best: Option[ContentType] = None
    var known: Option[ContentType] = None
    val chars = UCharacterIterator.getInstance(text)

    // loop over all tags to verify that everything is valid
    while (offs != UForwardCharacterIterator.DONE) {
      val next = (text.charAt(offs + 1) match {
        case '!' => text.charAt(offs + 2) match {
          case '[' =>
            // conditional sections are only allowed in XML
            known = Some(ContentType(ContentTypeCode.XML))
            chars.setIndex(offs + 2)
            XMLBreakIterator.matchedEndOfConditional(chars)
          case '-' if text.charAt(offs + 3) == '-' =>
            // comments default to XML
            best = best.orElse(Some(ContentType(ContentTypeCode.XML)))
            chars.setIndex(offs + 3)
            XMLBreakIterator.nextEndOfComment(chars)
          case _ =>
            // malformed XML, must be plain text
            known = Some(ContentType(ContentTypeCode.PlainText))
            UForwardCharacterIterator.DONE
        }
        case '?' =>
          // processing instruction sections are only allowed in XML
          known = Some(ContentType(ContentTypeCode.XML))
          chars.setIndex(offs + 2)
          XMLBreakIterator.nextEndOfPI(chars)
        case _ =>
          chars.setIndex(offs + (if (text.charAt(offs + 1) == '/') 2 else 1))

          val tagEnd = XMLBreakIterator.nextEndOfTag(chars)
          // check if the tag name matches one of the likely HTML tags
          ContentTypeDetector.getTagName(text, offs, tagEnd)
            .map(_.toLowerCase) match {
            case None =>
              known = Some(ContentType(ContentTypeCode.PlainText))
              UForwardCharacterIterator.DONE
            /* the list of HTML tags is intentionally non-exhaustive; it's
             * limited to just tags that require special content handling
             * that isn't handled by the XML tokenizer (<script> and <style>),
             * and the tags that would exist in effectively every meaningful
             * HTML document (the paragraph and line-break delimiters). */
            case Some("html") | Some("script") | Some("style") |
                Some("p") | Some ("br") =>
              best = Some(ContentType(ContentTypeCode.HTML))
              tagEnd
            case _ =>
              best = best.orElse(Some(ContentType(ContentTypeCode.XML)))
              tagEnd
          }
      })

      if (next == UForwardCharacterIterator.DONE || text.charAt(next - 1) != '>') {
        // invalid tag detected, fall back to plain-text and quit sniffing
        known = Some(ContentType(ContentTypeCode.PlainText))
        offs = UForwardCharacterIterator.DONE
      } else {
        // advance to the next tag
        offs = text.indexOf('<', next)
      }
    }
    known.orElse(best)
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

private[contenttype] object ContentTypeDetector {

  /** Tests if the character is a valid first character in an XML name
    *
    * {@link https://www.w3.org/TR/xml11/#NT-NameStartChar}
    */
  def isXmlNameStartChar(ch: Int) = {
    (ch == ':') || (ch >= 'A' && ch <= 'Z') || (ch == '_') ||
    (ch >= 'a' && ch <= 'z') || (ch >= 0xc0 && ch <= 0xd6) ||
    (ch >= 0xd8 && ch <= 0xf6) || (ch >= 0xf8 && ch <= 0x2ff) ||
    (ch >= 0x370 && ch <= 0x37d) || (ch >= 0x37f && ch <= 0x1fff) ||
    (ch >= 0x200c && ch <= 0x200d) || (ch >= 0x2070 && ch <= 0x218f) ||
    (ch >= 0x2c00 && ch <= 0x2fef) || (ch >= 0x3001 && ch <= 0xd7ff) ||
    /* NB: the surrogate blocks are actually not "valid" according to
     * the XML spec, but rather than checking for valid surrogate pairs
     * (since non-BMP characters are allowed in the spec), this code
     * just assumes that any high surrogate is part of a valid surrogate pair */
    (ch >= 0xd800 && ch <= 0xdbff) || (ch >= 0xf900 && ch <= 0xfdcf) ||
    (ch >= 0xfdf0 && ch <= 0xfffd)
  }

  /** Tests if the character is a valid character for an XML name
    *
    * {@link https://www.w3.org/TR/xml11/#NT-NameChar}
    */
  def isXmlNameChar(ch: Int): Boolean = {
    isXmlNameStartChar(ch) || (ch >= 0xdc00 && ch <= 0xdfff) ||
    (ch == '-') || (ch == '.') || (ch >= '0' && ch <= '9') ||
    (ch == 0xb7) || (ch >= 0x300 && ch <= 0x36f) ||
    (ch >= 0x203f && ch <= 0x2040)
  }

  def isXmlNameChar(ch: Char): Boolean = isXmlNameChar(ch.toInt)

  /** Returns the name of an XML tag within a text region
    *
    * If a valid tag exists between start and end, returns the name
    * of the tag.
    *
    * @param text text to analyze
    * @param start start of the text region
    * @param end end of the text region
    * @return the name of the tag if valid, else None
    */
  def getTagName(text: String, start: Int, end: Int): Option[String] = {
    if (end == UForwardCharacterIterator.DONE) return None
    // if the tag is an end tag, swallow the '/'
    val nameStart = start + (if (text.charAt(start + 1) == '/') 2 else 1)
    if (!isXmlNameStartChar(text.charAt(nameStart))) return None

    val name = (nameStart to end).toStream
      .map(text.charAt)
      .takeWhile(isXmlNameChar)
      .foldLeft("")((n: String, ch: Char) => n + ch)

    /* verify that the first character after the name is either whitespace,
     * a tag close, or an empty-tag marker */
    text.charAt(nameStart + name.length) match {
      case x if XMLBreakIterator.isWhitespace(x) => Some(name)
      case '/' | '>' => Some(name)
      case _ => None
    }
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
