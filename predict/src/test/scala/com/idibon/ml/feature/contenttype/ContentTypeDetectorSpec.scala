package com.idibon.ml.feature.contenttype

import scala.collection.mutable.HashMap

import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.alloy.{MemoryAlloyReader, MemoryAlloyWriter}

import org.json4s._
import org.scalatest.{Matchers, FunSpec}

/**
  * Created by haley on 3/9/16.
  */
class ContentTypeDetectorSpec extends FunSpec with Matchers {

  describe("save / load") {
    it("should default to shallow detection if no config block exists") {
      val transform = new ContentTypeDetectorLoader().load(new EmbeddedEngine, None, None)
      val document = JObject(List(
        JField("content", JString("<root>Deep detection will treat this as HTML</root><script></script>"))))
      transform(document) shouldBe ContentType(ContentTypeCode.PlainText)
    }

    it("should save and load correctly") {
      val deepArchive = HashMap[String, Array[Byte]]()
      val simpleArchive = HashMap[String, Array[Byte]]()
      val deepJson = new ContentTypeDetector(true).save(new MemoryAlloyWriter(deepArchive))
      val simpleJson = new ContentTypeDetector(false).save(new MemoryAlloyWriter(simpleArchive))
      val loader = new ContentTypeDetectorLoader
      val simple = loader.load(new EmbeddedEngine,
        Some(new MemoryAlloyReader(simpleArchive.toMap)), simpleJson)
      val deep = loader.load(new EmbeddedEngine,
        Some(new MemoryAlloyReader(deepArchive.toMap)), deepJson)

      val document = JObject(List(
        JField("content", JString("<root>Deep detection will treat this as HTML</root><script></script>"))))
      simple(document) shouldBe ContentType(ContentTypeCode.PlainText)
      deep(document) shouldBe ContentType(ContentTypeCode.HTML)
    }
  }

  describe("deepDetection = false") {

    val transform = new ContentTypeDetector(false)

    it("should use starts with tag detection when applicable") {
      val plainDoc = JObject(List(
        JField("content", JString("here is some cool plain text"))))
      val htmlDoc = JObject(List(
        JField("content", JString("<!DOCTYPE html> here is some text for an <strong> HTML </strong> document."))))
      val xmlDoc = JObject(List(
        JField("content", JString("<?xml version='1.0'?> xml what what"))))


      transform(plainDoc) shouldBe ContentType(ContentTypeCode.PlainText)
      transform(htmlDoc) shouldBe ContentType(ContentTypeCode.HTML)
      transform(xmlDoc) shouldBe ContentType(ContentTypeCode.XML)
    }

    it("should use .metadata.<sourcetype> if present") {
      val lexisDoc = JObject(List(
        JField("content", JString("here's some stuff from lexisnexis that <strong> probably </strong> has html")),
        JField("metadata", JObject(List(
          JField("lexisnexis", JString("lexisnexis stuff"))
        )))))
      val newscredDoc = JObject(List(
        JField("content", JString("here's some stuff from <strong> newscred </strong> with html")),
        JField("metadata", JObject(List(
          JField("newscred", JString("newscred stuff"))
        )))))
      val randomDoc = JObject(List(
        JField("content", JString("here's a random document with a metadata field")),
        JField("metadata", JObject(List(
          JField("random", JString("random source"))
        )))))

      transform(lexisDoc) shouldBe ContentType(ContentTypeCode.HTML)
      transform(newscredDoc) shouldBe ContentType(ContentTypeCode.HTML)
      transform(randomDoc) shouldBe ContentType(ContentTypeCode.PlainText)
    }
  }

  describe("deepDetection = true") {

    val transform = new ContentTypeDetector(true)

    it("should return plaintext if the last tag is truncated") {
      val broken = JObject(List(JField("content", JString("<html>foo<"))))
      transform(broken) shouldBe ContentType(ContentTypeCode.PlainText)
    }

    it("should return plaintext for empty strings") {
      val docEmpty = JObject(List(JField("content", JString(""))))
      transform(docEmpty) shouldBe ContentType(ContentTypeCode.PlainText)
    }

    it("should return plaintext if tags are invalid") {
      val docPlain = JObject(List(
        JField("content",
          JString("In HTML, all tags (like <p>) are represented by < and > characters"))))
      val docHTML = JObject(List(
        JField("content",
          JString("In HTML, all tags (like <p>) are represented by &lt; and &gt; characters"))))
      transform(docPlain) shouldBe ContentType(ContentTypeCode.PlainText)
      transform(docHTML) shouldBe ContentType(ContentTypeCode.HTML)
    }

    it("should default markup to XML unless specific HTML tags are recognized") {
      val docXML = JObject(List(
        JField("content", JString("<!-- comments --><doc>Some XML text</doc>"))))
      val docHTML = JObject(List(
        JField("content", JString("""<!-- comments --><script type="text/javascript">Some XML text</script>"""))))
      transform(docXML) shouldBe ContentType(ContentTypeCode.XML)
      transform(docHTML) shouldBe ContentType(ContentTypeCode.HTML)
    }

    it("should return XML if XML-only markup exists") {
      val docXML = JObject(List(
        JField("content",
          JString("""<?xslt-stylesheet ref="foo"?><p>XML document</p>"""))))
      val docXML2 = JObject(List(
        JField("content", JString("<p>XML document</p><?xslt-stylesheet ?>"))))
      val docXML3 = JObject(List(
        JField("content", JString("<![%foo; [ <![%bar; [ ]]> ]]><script></script>"))))
      transform(docXML) shouldBe ContentType(ContentTypeCode.XML)
      transform(docXML2) shouldBe ContentType(ContentTypeCode.XML)
      transform(docXML3) shouldBe ContentType(ContentTypeCode.XML)
    }

    it("should return plaintext for invalid markup") {
      val docInvalid = JObject(List(JField("content", JString("<!/foo>test"))))
      transform(docInvalid) shouldBe ContentType(ContentTypeCode.PlainText)
    }
  }
}
