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
      transform.apply(Seq[Token]()) shouldBe empty
    }

    it("should work on a sequence of Tokens") {
      val tokens = Seq[Feature[Token]](
        new Token("shaPes", Tag.Word, 0, 1), new Token("Sha.Pes", Tag.Word, 1, 1),
        new Token("Sha. Pes", Tag.Word, 2, 1), new Token("SHA8pes", Tag.Word, 3, 1))
      val expected = Seq[Feature[Shape]](
        new Shape("ccCcc", 1), new Shape("CccpCcc", 1), new Shape("CccpsCcc",1), new Shape("CCncc",1)
      )
      transform.apply(tokens) shouldBe expected
    }

    it("should only return the unique tokens with their occurrence counts") {
      val tokens = Seq[Feature[Token]](
        new Token("shaPes", Tag.Word, 0, 1), new Token("Sha.Pes", Tag.Word, 1, 1),
        new Token("Sha!,Pes", Tag.Word, 2, 1), new Token("SHA8pes", Tag.Word, 3, 1))
      val expected = Seq[Feature[Shape]](
        new Shape("ccCcc", 1), new Shape("CccpCcc", 2), new Shape("CCncc",1)
      )
      transform.apply(tokens) shouldBe expected
    }
  }
}
