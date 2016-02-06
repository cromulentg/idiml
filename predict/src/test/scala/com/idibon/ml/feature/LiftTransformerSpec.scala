package com.idibon.ml.feature

import com.idibon.ml.feature.wordshapes.Shape
import org.scalatest.{BeforeAndAfterAll, FunSpec, Matchers, BeforeAndAfter}

/**
  * Tests the LiftTransformer
  */
class LiftTransformerSpec extends FunSpec with Matchers
  with BeforeAndAfter with BeforeAndAfterAll {

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
