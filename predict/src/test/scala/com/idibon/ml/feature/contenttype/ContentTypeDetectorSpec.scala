package com.idibon.ml.feature.contenttype

import org.json4s._
import org.scalatest.{Matchers, FunSpec}

/**
  * Created by haley on 3/9/16.
  */
class ContentTypeDetectorSpec extends FunSpec with Matchers {

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
}
