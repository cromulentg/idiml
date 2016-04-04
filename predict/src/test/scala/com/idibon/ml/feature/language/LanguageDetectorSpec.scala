package com.idibon.ml.feature.language

import com.idibon.ml.feature.Feature
import com.idibon.ml.cld2.CLD2.DocumentMode
import com.idibon.ml.feature.contenttype.{ContentType, ContentTypeCode}

import org.json4s._
import org.json4s.JsonDSL._
import org.scalatest.{Matchers, FunSpec}

class LanguageDetectorSpec extends FunSpec with Matchers {

  describe("apply") {

    val transform = new LanguageDetector

    it("should use .metadata.iso_639_1 if present") {

      val document = JObject(List(
        JField("content", JString("This is an English document")),
        JField("metadata", JObject(List(
          JField("iso_639_1", JString("zh-Hant"))
        )))))
      transform(document, ContentType(ContentTypeCode.PlainText)) shouldBe LanguageCode(Some("zho"))
    }

    it("should use auto-detection if no metadata is present") {
      val document = JObject(List(
        JField("content", JString("Ceci est une phrase en français"))))
      transform(document, ContentType(ContentTypeCode.PlainText)) shouldBe LanguageCode(Some("fra"))
    }

    it("should ignore markup in HTML and XML mode") {
      val document = ("content" -> """<div class="class names are often english text">est une phrase en français</div>""")
      transform(document, ContentType(ContentTypeCode.PlainText)) shouldBe LanguageCode(Some("eng"))
      transform(document, ContentType(ContentTypeCode.HTML)) shouldBe LanguageCode(Some("fra"))
    }
  }

  describe("modeForContent") {
    it("should return the correct modes") {
      val transform = new LanguageDetector
      transform.modeForContent(ContentType(ContentTypeCode.PlainText)) shouldBe DocumentMode.PlainText
      transform.modeForContent(ContentType(ContentTypeCode.HTML)) shouldBe DocumentMode.HTML
      transform.modeForContent(ContentType(ContentTypeCode.XML)) shouldBe DocumentMode.HTML
    }
  }

  describe("normalize") {
    it("should return None if the language is invalid") {
      LanguageCode.normalize("!!") shouldBe None
    }
  }
}
