package com.idibon.ml.feature.tokenizer

import com.idibon.ml.feature.contenttype.ContentTypeCode

import com.ibm.icu.util.ULocale
import com.ibm.icu.text.BreakIterator
import org.scalatest.{Matchers, FunSpec}

class ICUTokenizerSpec extends FunSpec with Matchers {

  /** Returns tokenized content, minus whitespace */
  def !!(x: String, l: ULocale = ULocale.US,
      c: ContentTypeCode.Value = ContentTypeCode.PlainText) = {
    ICUTokenizer.tokenize(x, c, l)
      .filter(_.tag != Tag.Whitespace)
      .map(_.content)
  }

  /** Returns tokenized content, including whitespace */
  def ??(x: String, l: ULocale = ULocale.US,
      c: ContentTypeCode.Value = ContentTypeCode.PlainText) = {
    ICUTokenizer.tokenize(x, c, l).map(_.content)
  }

  /** Returns tuples of start and length for all tokens */
  def <<(x: String, l: ULocale = ULocale.US,
      c: ContentTypeCode.Value = ContentTypeCode.PlainText) = {
    ICUTokenizer.tokenize(x, c, l).map(t => (t.offset, t.length))
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

  describe("XML") {
    it("should treat XML markup as individual tokens") {
      !!("""<div class="foo">words</div>""", ULocale.US, ContentTypeCode.XML) shouldBe List(
        """<div class="foo">""", "words", "</div>")
    }

    it("should nest emoticons & HTML entity tokenization inside XML") {
      !!("<!DOCTYPE xml 'fo\">o'><h:p>words &amp; :)<test /></h:p>",
        ULocale.US, ContentTypeCode.XML) shouldBe List(
        "<!DOCTYPE xml 'fo\">o'>", "<h:p>", "words", "&amp;", ":)", "<test />", "</h:p>")
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

    it("should tokenize emoticons and character references") {
      !!("&amp; :):-)w00t!(/◕ヮ◕)/") shouldBe List("&amp;", ":)", ":-)", "w00t", "!", "(/◕ヮ◕)/")
      !!(" (╯°□°）╯︵ ┻━┻") shouldBe List("(╯°□°）╯","︵", "┻━┻")
      !!("༼ つ ಥ_ಥ ༽つ ノ( º _ ºノ)") shouldBe List("༼ つ ಥ_ಥ ༽つ", "ノ( º _ ºノ)")
    }

    it("should tokenize URLs") {
      ??("https://www.reddit.com/r/conspiracies is hilarious") shouldBe List(
        "https://www.reddit.com/r/conspiracies", " ", "is", " ", "hilarious")
      !!("check this out: http://t.co/Dwv1VnpJr0") shouldBe List("check", "this",
        "out", ":", "http://t.co/Dwv1VnpJr0")
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

  describe("thai") {
    it("should tokenize thai reasonably") {
      !!("สวัสดีมันเป็นสิ่งที่ดีที่ได้พบคุณ", ULocale.forLanguageTag("tha")) shouldBe List(
        "สวัสดี", "มัน", "เป็น", "สิ่ง", "ที่", "ดี", "ที่", "ได้", "พบ", "คุณ"
      )
    }
  }

  describe("japanese") {

    ignore("should group full-width numbers") {
      !!("１２３４５", ULocale.JAPAN) shouldBe List("１２３４５")
    }
  }

  describe("surrogate pairs") {

    it("should account for surrogate pairs in returned locations") {
      <<("happy \ud83d\ude00\ud83d\udc31") shouldBe
        List((0,5), (5, 1), (6, 2), (8, 2))
    }
  }

  describe("breaking") {

    it("should cache break iterators per locale") {
      val it = ICUTokenizer.breaking(ULocale.US, (b: BreakIterator) => b)
      ICUTokenizer.breaking(ULocale.US, (b: BreakIterator) => b) shouldBe theSameInstanceAs(it)
      ICUTokenizer.breaking(ULocale.JAPAN, (b: BreakIterator) => b) shouldNot be theSameInstanceAs(it)
    }

    it("should create multiple break iterators as-needed") {
      // create 3 threads to test contended usage
      val iterators = (1 to 3).toList.par.map(i => {
        ICUTokenizer.breaking(ULocale.US, (b: BreakIterator) => {
          Thread.sleep(250)
          b
        })
      })

      for (it <- iterators.tail)
        iterators(0) shouldNot be theSameInstanceAs(it)
    }
  }
}
