package com.idibon.ml.unicode

import org.scalatest.{Matchers, FunSpec}

/**
  * Created by haley on 3/28/16.
  */

class UnicodeConverterSpec extends FunSpec with Matchers {

  val extractor = new UnicodeConverter()

  describe("toIdiml") {
    //majority of the tests in here are adapted from the java-client unicode extractor tests
    it("should correctly handle out of bounds indices") {
      val content = "This is a document."
      UnicodeConverter.toIdiml(content, 19, 1) shouldBe (19,0)
      UnicodeConverter.toIdiml(content, 1, 19) shouldBe (1,18)
      UnicodeConverter.toIdiml(content, -5, 5) shouldBe (0,0)
      UnicodeConverter.toIdiml(content, -1, 8) shouldBe (0,7)
      UnicodeConverter.toIdiml(content, 16, 4) shouldBe (16,3)
      UnicodeConverter.toIdiml(content, -10, 100) shouldBe (0,19)
      UnicodeConverter.toIdiml(content, 0, -1) shouldBe (0,0)
    }

    it("should correctly handle non smp content") {
      val content = "Here is some normal doc content"
      UnicodeConverter.toIdiml(content, 0, 7) shouldBe(0, 7)
      UnicodeConverter.toIdiml(content, 9, 7) shouldBe(9, 7)
      UnicodeConverter.toIdiml(content, 20, 8) shouldBe(20, 8)
    }

    it("should correctly handle single smp content") {
      val content = "before \ud83d\udca9 after"
      UnicodeConverter.toIdiml(content, 0, 5) shouldBe(0, 5)
      UnicodeConverter.toIdiml(content, 3, 9) shouldBe(3, 10)
      UnicodeConverter.toIdiml(content, 9, 12) shouldBe(10, 5)
      UnicodeConverter.toIdiml(content, 7, 1) shouldBe(7, 2)
      UnicodeConverter.toIdiml(content, -10, 18) shouldBe(0, 9)
    }

    it("should correctly handle lots of smp content") {
      val content = "\ud83d\udc68+\ud83c\udf63=\ud83d\udca9"
      UnicodeConverter.toIdiml(content, 0, 2) shouldBe(0, 3)
      UnicodeConverter.toIdiml(content, 0, 1) shouldBe(0, 2)
      UnicodeConverter.toIdiml(content, 0, 3) shouldBe(0, 5)
      UnicodeConverter.toIdiml(content, 0, 5) shouldBe(0, 8)
      UnicodeConverter.toIdiml(content, 4, 1) shouldBe(6, 2)
      UnicodeConverter.toIdiml(content, 1, 3) shouldBe(2, 4)
    }

    it("should correctly handle broken utf16") {
      val content = "Broken: \ud83d\ufffd!"
      UnicodeConverter.toIdiml(content, 0, 8) shouldBe(0, 8)
      UnicodeConverter.toIdiml(content, 8, 1) shouldBe(8, 1)
      UnicodeConverter.toIdiml(content, 8, 2) shouldBe(8, 2)
      UnicodeConverter.toIdiml(content, 9, 2) shouldBe(9, 2)
    }
  }

  describe("toIdidat") {
   it("should correctly handle out of bounds indices") {
     val content = "This is a document."
     UnicodeConverter.toIdidat(content, 19, 1) shouldBe (19,0)
     UnicodeConverter.toIdidat(content, 1, 19) shouldBe (1,18)
     UnicodeConverter.toIdidat(content, -5, 5) shouldBe (0,0)
     UnicodeConverter.toIdidat(content, -1, 8) shouldBe (0,7)
     UnicodeConverter.toIdidat(content, 16, 4) shouldBe (16,3)
     UnicodeConverter.toIdidat(content, -10, 100) shouldBe (0,19)
     UnicodeConverter.toIdidat(content, 0, -1) shouldBe (0,0)
   }

    it("should correctly handle non smp content") {
      val content = "Here is some normal doc content"
      UnicodeConverter.toIdidat(content, 0, 7) shouldBe(0, 7)
      UnicodeConverter.toIdidat(content, 9, 7) shouldBe(9, 7)
      UnicodeConverter.toIdidat(content, 20, 8) shouldBe(20, 8)
    }

    it("should correctly handle single smp content") {
      val content = "before \ud83d\udca9 after"
      UnicodeConverter.toIdidat(content, 0, 5) shouldBe(0, 5)
      UnicodeConverter.toIdidat(content, 3, 9) shouldBe(3, 8)
      UnicodeConverter.toIdidat(content, 9, 12) shouldBe(8, 6)
      UnicodeConverter.toIdidat(content, 7, 2) shouldBe(7, 1)
    }

    it("should correctly handle lots of smp content") {
      val content = "\ud83d\udc68+\ud83c\udf63=\ud83d\udca9"
      UnicodeConverter.toIdidat(content, 0, 2) shouldBe(0, 1)
      UnicodeConverter.toIdidat(content, 0, 1) shouldBe(0, 1)
      UnicodeConverter.toIdidat(content, 0, 3) shouldBe(0, 2)
      UnicodeConverter.toIdidat(content, 0, 5) shouldBe(0, 3)
      UnicodeConverter.toIdidat(content, 3, 5) shouldBe(2, 3)
      UnicodeConverter.toIdidat(content, 2, 3) shouldBe(1, 2)
      UnicodeConverter.toIdidat(content, -10, 100) shouldBe(0, 5)
    }

    it("should correctly handle broken utf16") {
      val content = "Broken: \ud83d\ufffd!"
      UnicodeConverter.toIdidat(content, 0, 8) shouldBe(0, 8)
      UnicodeConverter.toIdidat(content, 8, 1) shouldBe(8, 1)
      UnicodeConverter.toIdidat(content, 8, 2) shouldBe(8, 2)
      UnicodeConverter.toIdidat(content, 9, 2) shouldBe(9, 2)
    }
  }
}
