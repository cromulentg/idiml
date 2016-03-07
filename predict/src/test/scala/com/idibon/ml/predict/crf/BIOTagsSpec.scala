package com.idibon.ml.predict.crf

import org.scalatest.{Matchers, FunSpec}

class BIOTagsSpec extends FunSpec with Matchers {

  describe("parse") {

    it("should split strings into the tag and the label") {
      BIOTag("O") shouldBe BIOOutside
      BIOTag("B1") shouldBe BIOLabel(BIOType.BEGIN, "1")
      BIOTag("I0") shouldBe BIOLabel(BIOType.INSIDE, "0")
    }

    it("should raise an exception if the tag is invalid") {
      intercept[IllegalArgumentException] { BIOTag("I") }
      intercept[IllegalArgumentException] { BIOTag("foobar") }
      intercept[IllegalArgumentException] { BIOTag("OUTSIDE") }
      intercept[IllegalArgumentException] { BIOTag("B") }
    }
  }

  describe("BIOLabel") {
    it("should prevent outside labels from being created") {
      intercept[IllegalArgumentException] {
        BIOLabel(BIOType.OUTSIDE, "foo")
      }
    }
  }

  describe("toString + parse") {
    it("should parse stringified tags") {
      val tests = List(BIOLabel(BIOType.BEGIN, "foo"),
        BIOLabel(BIOType.INSIDE, "bar"),
        BIOOutside)
      tests.foreach(x => BIOTag(x.toString) shouldBe x)
    }
  }
}
