package com.idibon.ml.feature.skipgram

import com.idibon.ml.feature.{StringFeature, ProductFeature}
import org.scalatest.{Matchers, FunSpec}
import com.idibon.ml.feature.tokenizer.{Token}


class SkipgramTransformerSpec extends FunSpec with Matchers{
  val transform = new SkipgramTransformer(2, 3)
  val transform2 = new SkipgramTransformer(3,0)
  val transform3 = new SkipgramTransformer(2,3)
  val transform4 = new SkipgramTransformer(0,3)
  val transform5 = new SkipgramTransformer(2,7)

  val tokens = Vector(StringFeature("Insurgents"), StringFeature("killed"),
    StringFeature("in"), StringFeature("ongoing"),StringFeature("fighting"))

  describe("getSkipPermutations"){
    it("there should be none if n is 0") {
      transform2.getSkipPermutations() shouldBe empty
    }

    it("should should generate the correct permutations") {
      transform3.getSkipPermutations() shouldBe List(
        List(0, 0), List(0, 1), List(1, 0),
        List(0, 2), List(2, 0), List(1, 1),
        List(1, 2), List(2, 1), List(2, 2)
      )
    }
  }

  describe("apply") {

    it("should work on an empty sequence") {
      transform(Seq[Token]()) shouldBe empty
    }

    it("should return the empty set when n=0"){
      transform2(tokens) shouldBe empty
    }

    it("should return the empty set when min(n,k) is larger than the input"){
      transform5(tokens) shouldBe empty
    }

    it("should act like ngrams when k=0"){
      //tri-grams
      transform4(tokens) shouldBe List(
        ProductFeature(List(tokens(0), tokens(1), tokens(2))),
        ProductFeature(List(tokens(1), tokens(2), tokens(3))),
        ProductFeature(List(tokens(2), tokens(3), tokens(4)))
      )
    }

    it("should generate a correct sequence of skipgrams") {
      //2-skip-tri-grams
      transform(tokens) shouldBe List(
        ProductFeature(List(tokens(0), tokens(1), tokens(2))),
        ProductFeature(List(tokens(0), tokens(1), tokens(3))),
        ProductFeature(List(tokens(0), tokens(2), tokens(3))),
        ProductFeature(List(tokens(0), tokens(1), tokens(4))),
        ProductFeature(List(tokens(0), tokens(3), tokens(4))),
        ProductFeature(List(tokens(0), tokens(2), tokens(4))),
        ProductFeature(List(tokens(1), tokens(2), tokens(3))),
        ProductFeature(List(tokens(1), tokens(2), tokens(4))),
        ProductFeature(List(tokens(1), tokens(3), tokens(4))),
        ProductFeature(List(tokens(2), tokens(3), tokens(4)))
      )
    }
  }
}
