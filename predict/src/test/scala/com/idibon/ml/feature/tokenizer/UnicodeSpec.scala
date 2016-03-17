package com.idibon.ml.feature.tokenizer

import com.idibon.ml.feature.tokenizer.Unicode._

import org.scalatest.{Matchers, FunSpec}

class UnicodeSpec extends FunSpec with Matchers {

  it("should detect unicode punctuation") {
    // various codepoints in Punctuation, Other category (Po)
    List(0x21, 0x2e, 0x2f, 0x40, 0xbf, 0x55a, 0x831, 0x104a, 0x16ed,
      0x1b5b, 0x2031, 0xa9c9, 0xfe50, 0x10af4, 0x111dd).foreach(v => {
        (v -> isPunctuation(v)) shouldBe (v -> true)
      })
    // various codepoints in Punctuation, Dash category (Pd)
    List(0x2d, 0x5be, 0x2014, 0x301c, 0x30a0, 0xfe32, 0xff0d).foreach(v => {
      (v -> isPunctuation(v)) shouldBe (v -> true)
    })
    // "" Punctuation, Close (Pe)
    List(0x29, 0x5d, 0x29d9, 0x3017, 0xfe44, 0xff63).foreach(v => {
      (v -> isPunctuation(v)) shouldBe (v -> true)
    })
    // "" Punctuation, Final Quote (Pf)
    List(0xbb, 0x2e03, 0x2e21).foreach(v => {
      (v -> isPunctuation(v)) shouldBe (v -> true)
    })
    // "" Punctuation, Initial quote (Pi)
    List(0xab, 0x2018, 0x2e20).foreach(v => {
      (v -> isPunctuation(v)) shouldBe (v -> true)
    })
    // "" Punctuation, Open (Ps)
    List(0x5b, 0x7b, 0x201e, 0x2772, 0x300c, 0xfe5d).foreach(v => {
      (v -> isPunctuation(v)) shouldBe (v -> true)
    })
    // Symbol, Currency (Sc)
    List(0x24, 0xa2, 0x9f2, 0xaf1, 0xbf9, 0x20b2, 0xffe6).foreach(v => {
      (v -> isPunctuation(v)) shouldBe (v -> false)
    })
  }

  it("should detect combining marks as extenders") {
    // Combining Diacritical Marks block (0x300 - 0x36f)
    (0x300 to 0x36f).foreach(v => isGraphemeExtender(v) shouldBe true)
    // various codepoints in the Mark, Enclosing category (Me)
    List(0x488, 0x489, 0x1abe, 0x20e0, 0xa672).foreach(v => {
      (v -> isGraphemeExtender(v)) shouldBe (v -> true)
    })
    // various codepoints in Mark, Nonspacing category (Mn)
    List(0x484, 0x591, 0x5a0, 0x5b1, 0x64c, 0x711, 0x746, 0x9c1,
      0xd43, 0x17bb, 0xa8c4, 0x0fe02, 0x101fd, 0x11131, 0x1da1e).foreach(v => {
        (v -> isGraphemeExtender(v)) shouldBe (v -> true)
      })
    // codepoints in Mark, Spacing Combining (Mc) with GRAPHEME_EXTEND property
    List(0xcd5, 0xdcf, 0x1d165, 0x1d171, 0x1e8d2, 0xe0101).foreach(v => {
      (v -> isGraphemeExtender(v)) shouldBe (v -> true)
    })
    // various codepoints in Mark, Spacing Combining category (Mc)
    List(0x903, 0x93f, 0x94e, 0xa3e, 0xcbe, 0xdde, 0x1a61, 0x1ce1,
      0xaaef, 0x16f70).foreach(v => {
        (v -> isGraphemeExtender(v)) shouldBe (v -> false)
      })
  }
}
