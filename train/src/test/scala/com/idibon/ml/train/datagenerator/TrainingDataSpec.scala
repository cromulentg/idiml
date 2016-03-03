package com.idibon.ml.train.datagenerator.json

import org.scalatest.{Matchers, FunSpec}
import org.json4s.native.JsonMethods.parse
import org.json4s._

class TrainingDataSpec extends FunSpec with Matchers {

  describe("JSON") {
    implicit val formats = org.json4s.DefaultFormats

    it("should read documents with empty annotations") {
      val json = parse("""
{"name":"document","content":"Some content","metadata":{},
"annotations":[]}""")

      val doc = json.extract[Document]

      doc.annotations shouldBe empty
      doc.content shouldBe "Some content"
    }

    it("should read documents with span annotations") {
      val json = parse("""
{"name":"document","content":"Span annotated document",
"annotations":[{"label":{"name":"NER"},"isPositive":true,"offset":0,"length":4}]}""")
      val doc = json.extract[Document]
      doc.annotations shouldBe List(Annotation(LabelName("NER"), true, Some(0), Some(4)))
      doc.annotations.forall(_.isSpan) shouldBe true
    }

    it("should read documents with document annotations") {
      val json = parse("""
{"name":"document","content":"Span annotated document",
"annotations":[{"label":{"name":"positive"},"isPositive":true},
{"label":{"name":"negative"},"isPositive":false}]}""")
      val doc = json.extract[Document]
      doc.annotations shouldBe List(Annotation(LabelName("positive"), true, None, None),
        Annotation(LabelName("negative"), false, None, None))
      doc.annotations.exists(_.isSpan) shouldBe false
    }
  }

  describe("Annotation#inside") {

    it("should return true iff the point is inside the span") {
      val ann = Annotation(LabelName("test"), true, Some(2), Some(5))
      (2 until 7).foreach(i => ann.inside(i) shouldBe true)
      (0 until 2).foreach(i => ann.inside(i) shouldBe false)
      ann.inside(7) shouldBe false
    }
  }
}
