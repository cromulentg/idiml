package com.idibon.ml.feature.contenttype

import com.idibon.ml.feature._
import com.idibon.ml.alloy.Codec

/** Identifies content types that benefit from special handling.
  *
  * NB: Add new values to the end of the list!
  */
object ContentTypeCode extends Enumeration {
  val PlainText,
    HTML,
    XML = Value
}

/** Feature representing a document's content type */
case class ContentType(code: ContentTypeCode.Value) extends Feature[ContentType]
    with Buildable[ContentType, ContentTypeBuilder] {

  def get = this

  def save(output: FeatureOutputStream) {
    Codec.VLuint.write(output, code.id)
  }

  def getAsString: Option[String] = {
    val contentTypeCodeString = this.code match {
      case ContentTypeCode.HTML => "HTML"
      case ContentTypeCode.PlainText => "PlainText"
      case ContentTypeCode.XML => "XML"
    }
    Some(contentTypeCodeString)
  }
}

class ContentTypeBuilder extends Builder[ContentType] {
  def build(input: FeatureInputStream) = {
    ContentType(ContentTypeCode(Codec.VLuint.read(input)))
  }
}
