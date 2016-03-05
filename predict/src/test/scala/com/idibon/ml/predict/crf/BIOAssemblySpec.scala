package com.idibon.ml.predict.crf

import com.idibon.ml.feature.language.LanguageCode
import com.idibon.ml.feature.tokenizer.{TokenTransformer, Tag}
import com.idibon.ml.feature.contenttype.{ContentType, ContentTypeCode}
import com.idibon.ml.feature.StringFeature
import com.idibon.ml.predict.Span

import com.ibm.icu.util.ULocale
import org.scalatest.{Matchers, FunSpec}

class BIOAssemblySpec extends FunSpec with Matchers with BIOAssembly {

  val tokenizer = new TokenTransformer()

  def tokenize(x: String) = tokenizer(StringFeature(x), LanguageCode(Some("eng")),
    ContentType(ContentTypeCode.PlainText)).filter(_.tag != Tag.Whitespace)

  def tag(x: String*) = x map BIOTag

  it("should support empty lists") {
    assemble(Seq(), Seq()) shouldBe Seq()
  }

  it("should return an empty list if all tokens are outside") {
    val tokens = tokenize("hello, world")
    assemble(tokenize("hello, world"), tag("O", "O", "O")) shouldBe Seq()
  }

  it("should skip over invalid INSIDE tags") {
    assemble(tokenize("hello, world"), tag("I0", "I0", "B0")) shouldBe Seq(
      Span("0", 1.0f, 0, 7, 5))
  }

  it("should return single-token spans") {
    assemble(tokenize("hello, world"), tag("B0", "B0", "B1")) shouldBe Seq(
      Span("0", 1.0f, 0, 0, 5),
      Span("0", 1.0f, 0, 5, 1),
      Span("1", 1.0f, 0, 7, 5)
    )
  }

  it("should assemble complex spans") {
    assemble(tokenize("the quick brown fox jumped over the lazy dog"),
      tag("O", "BNP", "INP", "INP", "BV", "O", "O", "BNP", "INP")) shouldBe Seq(
        Span("NP", 1.0f, 0, 4, 15),
        Span("V", 1.0f, 0, 20, 6),
        Span("NP", 1.0f, 0, 36, 8))
  }

  it("should die if the lists have different lengths") {
    intercept[IllegalArgumentException] {
      assemble(tokenize("hello, world"), tag("O"))
    }
  }
}
