package com.idibon.ml.feature.wordshapes

import com.idibon.ml.feature.Feature
import com.idibon.ml.feature.tokenizer.{Tag, Token}
import org.scalatest.{Matchers, BeforeAndAfter, FunSpec}

class WordShapesSpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("WordShapes") {

    var transform: WordShapesTransformer = null

    before {
      transform = new WordShapesTransformer()
    }

    it("should work on an empty sequence") {
      transform(Seq[Token]()) shouldBe empty
    }

    it("should work on a sequence of tokens") {
      val tokens = Seq[Feature[Token]](
        new Token("shaPes", Tag.Word, 0, 1), new Token("Sha.Pes", Tag.Word, 1, 1),
        new Token("Sha!,Pes", Tag.Word, 2, 1), new Token("SHA8pes", Tag.Word, 3, 1))
      val expected = Seq[Feature[Shape]](
        new Shape("ccCcc"), new Shape("CccpCcc"), new Shape("CccpCcc"), new Shape("CCncc")
      )
      transform(tokens) shouldBe expected
    }

    it("should have unicode support") {
      val tokens = Seq[Feature[Token]](
        new Token("Mañana", Tag.Word, 1, 1),
        new Token("\uD83D\uDE00  \uD83D\uDE02", Tag.Word, 2, 1) //laughing emoji + crying laughing emoji
      )

      val expected = Seq[Feature[Shape]](
        new Shape("Ccc"), new Shape("psp")
      )
      transform(tokens) shouldBe expected
    }

    it ("should return None from getAsString") {
      Shape("Cc").getAsString shouldBe None
    }

  }
}
