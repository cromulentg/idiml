package com.idibon.ml.feature

import com.idibon.ml.feature.bagofwords.Word
import com.idibon.ml.feature.wordshapes.Shape
import org.scalatest.{FunSpec, Matchers}

/**
  * Tests the LiftTransformer
  */
class LiftTransformerSpec extends FunSpec with Matchers {

  describe("it works as intended") {

    val transform = new LiftTransformer()
    it("works with one feature") {
      val feature = new Shape("bogusValue")
      transform.apply(feature) shouldBe Seq[Feature[Shape]](feature)
    }
    it("works with more than one feature") {
      val feature1 = new Shape("bogusValue1")
      val feature2 = new Shape("bogusValue2")
      val feature3 = new Shape("bogusValue3")
      transform.apply(feature1, feature2, feature3) shouldBe Seq[Feature[Shape]](feature1, feature2, feature3)
    }
    it("works with no features") {
      transform.apply() shouldBe Seq()
    }
  }
}

class ChainLiftTransformerSpec extends FunSpec with Matchers {

  it("should perform component-wise concatenation") {
    val lift = new ChainLiftTransformer()
    val a = Chain(Word("a"), Word("B"))
    val b = Chain(Shape("c"), Shape("C"))
    lift(a, b) shouldBe Chain(
      Seq[Feature[_]](Shape("c"), Word("a")),
      Seq[Feature[_]](Shape("C"), Word("B")))
  }
}
