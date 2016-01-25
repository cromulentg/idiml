package com.idibon.ml.feature.ngram

import org.scalatest.{Matchers, FunSpec}
import com.idibon.ml.feature.{Feature, ProductFeature, StringFeature}
import com.idibon.ml.feature.tokenizer.{Tag, Token}

class NgramTransformerSpec extends FunSpec with Matchers {

  describe("apply") {
    val transform = new NgramTransformer(1, 3)

    it("should work on an empty sequence") {
      transform(Seq[Token]()) shouldBe empty
    }

    it("should not return partial n-grams for short input sequences") {
      val tokens = Vector(Token("a", Tag.Word, 0, 1), Token("token", Tag.Word, 2, 5))
      transform(tokens) shouldBe List(
        ProductFeature(Seq(tokens(0), tokens(1))),
        ProductFeature(Seq(tokens(0))),
        ProductFeature(Seq(tokens(1))))
    }

    it("should generate overlapping sequences") {
      val features = Vector(StringFeature("a"), StringFeature("b"),
        StringFeature("c"), StringFeature("d"), StringFeature("e"))
      transform(features) shouldBe List(
        ProductFeature(Seq(features(0), features(1), features(2))),
        ProductFeature(Seq(features(1), features(2), features(3))),
        ProductFeature(Seq(features(2), features(3), features(4))),
        ProductFeature(Seq(features(0), features(1))),
        ProductFeature(Seq(features(1), features(2))),
        ProductFeature(Seq(features(2), features(3))),
        ProductFeature(Seq(features(3), features(4))),
        ProductFeature(Seq(features(0))),
        ProductFeature(Seq(features(1))),
        ProductFeature(Seq(features(2))),
        ProductFeature(Seq(features(3))),
        ProductFeature(Seq(features(4))))
    }
  }
}
