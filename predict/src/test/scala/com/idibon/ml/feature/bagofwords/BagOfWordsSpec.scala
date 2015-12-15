package com.idibon.ml.feature.bagofwords

import com.idibon.ml.feature.Feature
import com.idibon.ml.feature.tokenizer.{Tag, Token}
import org.scalatest.{Matchers, BeforeAndAfter, FunSpec}

class BagOfWordsSpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("BagOfWords") {

    var transform: BagOfWordsTransformer = null

    before {
      transform = new BagOfWordsTransformer()
    }

    it("should work on an empty sequence") {
      transform.apply(Seq[Token]()) shouldBe empty
    }

    it("should work on a sequence of Tokens") {
      val twoTokens = Seq[Feature[Token]](
        new Token("token", Tag.Word, 0, 1), new Token("words", Tag.Word, 1, 1))
      val expected = Seq[Feature[Word]](
        new Word("token", 1), new Word("words", 1)
      )
      transform.apply(twoTokens) shouldBe expected
    }

    it("should only return the unique tokens with their occurrence counts") {
      val twoTokens = Seq[Feature[Token]](
        new Token("token", Tag.Word, 0, 1), new Token("words", Tag.Word, 1, 1),
        new Token("words", Tag.Word, 2, 1))
      val expected = Seq[Feature[Word]](
        new Word("token", 1), new Word("words", 2)
      )
      transform.apply(twoTokens) shouldBe expected
    }
  }
}
