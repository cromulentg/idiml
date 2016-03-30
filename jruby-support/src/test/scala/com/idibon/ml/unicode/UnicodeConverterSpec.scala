package com.idibon.ml.unicode

import org.scalatest.{Matchers, FunSpec}

/**
  * Created by haley on 3/28/16.
  */

class UnicodeConverterSpec extends FunSpec with Matchers {

  val extractor = new UnicodeConverter()

  describe("from32to16") {
    //majority of the tests in here are adapted from the java-client unicode extractor tests

    it("should correctly handle out of bounds indices") {
      val content = "This is a document."
      UnicodeConverter.from32to16(content, 19, 1) shouldBe (19, 0)
      UnicodeConverter.from32to16(content, 1, 19) shouldBe (1, 18)
      UnicodeConverter.from32to16(content, -5, 5) shouldBe (0, 0)
      UnicodeConverter.from32to16(content, -1, 8) shouldBe (0, 7)
      UnicodeConverter.from32to16(content, 16, 4) shouldBe (16, 3)
      UnicodeConverter.from32to16(content, -10, 100) shouldBe (0, 19)
      UnicodeConverter.from32to16(content, 0, -1) shouldBe (0, 0)
    }

    it("should correctly handle bmp content") {
      val content = "Here is some normal doc content"
      UnicodeConverter.from32to16(content, 0, 7) shouldBe(0, 7)
      UnicodeConverter.from32to16(content, 9, 7) shouldBe(9, 7)
      UnicodeConverter.from32to16(content, 20, 8) shouldBe(20, 8)
    }

    it("should correctly handle single smp content") {
      val content = "before \ud83d\udca9 after"
      UnicodeConverter.from32to16(content, 0, 5) shouldBe(0, 5)
      UnicodeConverter.from32to16(content, 3, 9) shouldBe(3, 10)
      UnicodeConverter.from32to16(content, 9, 12) shouldBe(10, 5)
      UnicodeConverter.from32to16(content, 7, 1) shouldBe(7, 2)
      UnicodeConverter.from32to16(content, -10, 18) shouldBe(0, 9)
    }

    it("should correctly handle lots of smp content") {
      val content = "\ud83d\udc68+\ud83c\udf63=\ud83d\udca9"
      UnicodeConverter.from32to16(content, 0, 2) shouldBe(0, 3)
      UnicodeConverter.from32to16(content, 0, 1) shouldBe(0, 2)
      UnicodeConverter.from32to16(content, 0, 3) shouldBe(0, 5)
      UnicodeConverter.from32to16(content, 0, 5) shouldBe(0, 8)
      UnicodeConverter.from32to16(content, 4, 1) shouldBe(6, 2)
      UnicodeConverter.from32to16(content, 1, 3) shouldBe(2, 4)
    }

    it("should correctly handle broken utf16") {
      val content = "Broken: \ud83d\ufffd!"
      UnicodeConverter.from32to16(content, 0, 8) shouldBe(0, 8)
      UnicodeConverter.from32to16(content, 8, 1) shouldBe(8, 1)
      UnicodeConverter.from32to16(content, 8, 2) shouldBe(8, 2)
      UnicodeConverter.from32to16(content, 9, 2) shouldBe(9, 2)
    }
  }

  describe("from16to32") {

    it("should correctly handle out of bounds indices") {
      val content = "This is a document."
      UnicodeConverter.from16to32(content, 19, 1) shouldBe (19, 0)
      UnicodeConverter.from16to32(content, 1, 19) shouldBe (1, 18)
      UnicodeConverter.from16to32(content, -5, 5) shouldBe (0, 0)
      UnicodeConverter.from16to32(content, -1, 8) shouldBe (0, 7)
      UnicodeConverter.from16to32(content, 16, 4) shouldBe (16, 3)
      UnicodeConverter.from16to32(content, -10, 100) shouldBe (0, 19)
      UnicodeConverter.from16to32(content, 0, -1) shouldBe (0, 0)
    }

    it("should correctly handle bmp content") {
      val content = "Here is some normal doc content"
      UnicodeConverter.from16to32(content, 0, 7) shouldBe(0, 7)
      UnicodeConverter.from16to32(content, 9, 7) shouldBe(9, 7)
      UnicodeConverter.from16to32(content, 20, 8) shouldBe(20, 8)
    }

    it("should correctly handle single smp content") {
      val content = "before \ud83d\udca9 after"
      UnicodeConverter.from16to32(content, 0, 5) shouldBe(0, 5)
      UnicodeConverter.from16to32(content, 3, 9) shouldBe(3, 8)
      UnicodeConverter.from16to32(content, 9, 12) shouldBe(8, 6)
      UnicodeConverter.from16to32(content, 7, 2) shouldBe(7, 1)
    }

    it("should correctly handle lots of smp content") {
      val content = "\ud83d\udc68+\ud83c\udf63=\ud83d\udca9"
      UnicodeConverter.from16to32(content, 0, 2) shouldBe(0, 1)
      UnicodeConverter.from16to32(content, 0, 1) shouldBe(0, 1)
      UnicodeConverter.from16to32(content, 0, 3) shouldBe(0, 2)
      UnicodeConverter.from16to32(content, 0, 5) shouldBe(0, 3)
      UnicodeConverter.from16to32(content, 3, 5) shouldBe(2, 3)
      UnicodeConverter.from16to32(content, 2, 3) shouldBe(1, 2)
      UnicodeConverter.from16to32(content, -10, 100) shouldBe(0, 5)
    }

    it("should correctly handle broken utf16") {
      val content = "Broken: \ud83d\ufffd!"
      UnicodeConverter.from16to32(content, 0, 8) shouldBe(0, 8)
      UnicodeConverter.from16to32(content, 8, 1) shouldBe(8, 1)
      UnicodeConverter.from16to32(content, 8, 2) shouldBe(8, 2)
      UnicodeConverter.from16to32(content, 9, 2) shouldBe(9, 2)
    }
  }
}
