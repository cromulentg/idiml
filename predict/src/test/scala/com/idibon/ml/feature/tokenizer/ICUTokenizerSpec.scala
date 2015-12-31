package com.idibon.ml.feature.tokenizer

import com.ibm.icu.util.ULocale
import org.scalatest.{Matchers, FunSpec}

class ICUTokenizerSpec extends FunSpec with Matchers {

  /** Returns tokenized content, minus whitespace */
  def !!(x: String, l: ULocale = ULocale.US) = {
    ICUTokenizer.tokenize(x, l)
      .filter(_.tag != Tag.Whitespace)
      .map(_.content)
  }

  /** Returns tokenized content, including whitespace */
  def ??(x: String, l: ULocale = ULocale.US) = {
    ICUTokenizer.tokenize(x, l).map(_.content)
  }

  /** Returns tuples of start and length for all tokens */
  def <<(x: String, l: ULocale = ULocale.US) = {
    ICUTokenizer.tokenize(x, l).map(t => (t.offset, t.length))
  }

  describe("weird") {
    it("should group CRLF") {
      ??("\r\n") shouldBe List("\r\n")
      ??("\n\n") shouldBe List("\n", "\n")
      ??("\r\n\r\n") shouldBe List("\r\n", "\r\n")
    }

    it("should tokenize whitespace") {
      ??("\t  \t") shouldBe List("\t", " ", " ", "\t")
    }

    it("should handle empty strings") {
      ??("") shouldBe empty
    }
  }

  describe("english") {

    ignore("should tokenize possessives") {
      !!("Matt's stuff") shouldBe List("Matt", "'s", "stuff")
      !!("Players' Union") shouldBe List("Players", "'", "Union")
    }

    it("should group contractions") {
      !!("don't") shouldBe List("don't")
      !!("O'Henry") shouldBe List("O'Henry")
    }

    it("should group numeric values") {
      !!("1,000,000.09") shouldBe List("1,000,000.09")
    }

    ignore("should tokenize times of day") {
      !!("1:15PM") shouldBe List("1:15", "PM")
    }

    ignore("should tokenize currencies") {
      !!("$100USD") shouldBe List("$100", "USD")
    }

    it("should tokenize equations") {
      !!("1+1=2") shouldBe List("1", "+", "1", "=", "2")
    }

    it("should group abbreviations") {
      !!("U.S.A") shouldBe List("U.S.A")
      !!("u.s.a") shouldBe List("u.s.a")
    }

    ignore("should tokenize non-abbreviations") {
      !!("amper.sand") shouldBe List("amper", ".", "sand")
    }

    ignore("should group ampersands in abbreviations") {
      !!("AT&T") shouldBe List("AT&T")
      !!("at&t") shouldBe List("at&t")
    }

    it("should tokenize ampersands between words") {
      !!("W 1st St & 3rd") shouldBe List("W", "1st", "St", "&", "3rd")
      !!("W 1st St&3rd") shouldBe List("W", "1st", "St", "&", "3rd")
    }
  }

  describe("japanese") {

    ignore("should group full-width numbers") {
      !!("１２３４５", ULocale.JAPAN) shouldBe List("１２３４５")
    }
  }

  describe("surrogate pairs") {

    ignore("should account for surrogate pairs in returned locations") {
      <<("happy \ud83d\ude00\ud83d\udc31") shouldBe
        List((0,5), (5, 1), (6, 1), (7, 1))
    }
  }
}
