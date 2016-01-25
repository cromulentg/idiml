package com.idibon.ml.feature.bagofwords

import com.idibon.ml.feature.Feature
import com.idibon.ml.feature.language.LanguageCode
import com.idibon.ml.feature.tokenizer.{Tag, Token}
import org.scalatest.{Matchers, BeforeAndAfter, FunSpec}

class BagOfWordsSpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("accept-all, no case folding") {

    val transform = new BagOfWordsTransformer(Tag.values.toList, CaseFoldOp.None)

    it("should work on an empty sequence") {
      transform(Seq[Token](), LanguageCode(None)) shouldBe empty
    }

    it("should work on a sequence of Tokens") {
      val twoTokens = Seq[Feature[Token]](
        new Token("token", Tag.Word, 0, 1), new Token("words", Tag.Word, 1, 1))
      val expected = Seq[Feature[Word]](
        new Word("token"), new Word("words")
      )
      transform(twoTokens, LanguageCode(None)) shouldBe expected
    }

    it("should return every token") {
      val twoTokens = Seq[Feature[Token]](
        new Token("token", Tag.Word, 0, 1), new Token("words", Tag.Word, 1, 1),
        new Token("words", Tag.Word, 2, 1))
      val expected = Seq[Feature[Word]](
        new Word("token"), new Word("words"), new Word("words")
      )
      transform(twoTokens, LanguageCode(None)) shouldBe expected
    }
  }

  describe("only accept words, no case folding") {
    val transform = new BagOfWordsTransformer(List(Tag.Word), CaseFoldOp.None)
    val tokens = List(
      Token("this", Tag.Word, 0, 4), Token(" ", Tag.Whitespace, 4, 1),
      Token("is", Tag.Word, 5, 2), Token(" ", Tag.Whitespace, 7, 1),
      Token("a", Tag.Word, 8, 1), Token(" ", Tag.Whitespace, 9, 1),
      Token("token", Tag.Word, 10, 5), Token("!", Tag.Punctuation, 15, 1))
    val expected = List(new Word("this"), new Word("is"),
      new Word("a"), new Word("token"))
    transform(tokens, LanguageCode(None)) shouldBe expected
  }
}
