package com.idibon.ml.feature.tokenizer

import org.scalatest.{Matchers, FunSpec}

class TokenSpec extends FunSpec with Matchers {

  describe ("of") {

    it("should identify BMP Whitespace") {
      // http://www.fileformat.info/info/unicode/category/Zs/list.htm
      for (x <- List("\r\n \t", "\u00a0", "\u202f", "\u2000",
              "\u2001", "\u3000", "\u2009", "\u205f"))
        Tag.of(x) shouldBe Tag.Whitespace
    }

    it("should identify BMP Punctuation") {
      // most of these taken from unicode Punctuation, other:
      // http://www.fileformat.info/info/unicode/category/Po/list.htm
      for (x <- List(".,?!#-/\\@", "\uff0e", "\uff02", "\u0701",
              "\u061f", "\u0df4", "\u1801", "\u00a7", "\u055f",
              "\u060c", "\u2014")) {
        Tag.of(x) shouldBe Tag.Punctuation
      }
    }

    it("should identify mathematical symbols") {
      // FIXME: add a new category for symbols?
      for (x <- List("\u00f7", "+", "\u222b"))
        Tag.of(x) shouldBe Tag.Word
    }

    ignore("should identify SMP (surrogate-pair) Punctuation") {
      // FIXME: of() isn't surrogate-pair safe
      for (x <- List("\ud805\udf3d", "\ud836\ude8a"))
        Tag.of(x) shouldBe Tag.Punctuation
    }
  }
}
