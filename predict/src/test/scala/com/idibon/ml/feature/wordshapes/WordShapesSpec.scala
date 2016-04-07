package com.idibon.ml.feature.wordshapes

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import com.idibon.ml.feature.{Chain, Feature,
  FeatureInputStream, FeatureOutputStream}
import com.idibon.ml.feature.tokenizer.{Tag, Token}
import org.scalatest.{Matchers, BeforeAndAfter, FunSpec}

class WordShapesSpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("ChainOfWordShapes") {
    it("should generate chains") {
      val xf = new ChainWordShapesTransformer()
      xf(Chain(Token("hi", Tag.Word, 0, 2), Token("!", Tag.Punctuation, 3, 1))) shouldBe Chain(Shape("cc"), Shape("p"))
    }
  }

  describe("save / load") {
    it("should save and load word shapes features") {
      val shape = Shape("psp")
      val os = new ByteArrayOutputStream
      new FeatureOutputStream(os).writeFeature(shape)
      val loaded = new FeatureInputStream(new ByteArrayInputStream(os.toByteArray)).readFeature
      loaded shouldBe shape
    }
  }

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
  }

  describe("get") {

    it("should return None from getHumanReadableString") {
      Shape("Cc").getHumanReadableString shouldBe None
    }
  }

}
