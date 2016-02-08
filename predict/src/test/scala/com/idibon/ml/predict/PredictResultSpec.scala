package com.idibon.ml.predict

import org.scalatest.{Matchers, FunSpec}
import com.idibon.ml.feature.bagofwords.Word

class ClassificationSpec extends FunSpec with Matchers {

  describe("#average") {

    it("should raise an exception if you average multiple labels") {
      intercept[IllegalArgumentException] {
        Classification.average(Seq(
          Classification("foo", 1.0f, 1, 0, Seq()),
          Classification("bar", 1.0f, 1, 0, Seq())), (c) => c.matchCount)
      }
    }

    it("should return a 0.0 confidence if there is no weight returned") {
      val result = Classification.average(Seq(
        Classification("foo", 1.0f, 1, 0, Seq()),
        Classification("foo", 1.0f, 1, 0, Seq())), (c) => 0)
      result.label shouldBe "foo"
      result.probability shouldBe 0.0f
      result.matchCount shouldBe 0
    }

    it("should perform a weighted average of probabilities") {
      val result = Classification.average(Seq(
        Classification("foo", 1.0f, 2, 0, Seq()),
        Classification("foo", 0.0f, 1, 0, Seq()),
        Classification("foo", 0.0f, 1, 0, Seq())), (c) => c.matchCount)
      result.label shouldBe "foo"
      result.probability shouldBe 0.5f
      result.matchCount shouldBe 4
      result.isForced shouldBe false
    }

    it("should return the superset of flags and significant features") {
      val result = Classification.average(Seq(
        Classification("foo", 0.5f, 1, 0x5a5a, Seq(Word("bar") -> 0.5f)),
        Classification("foo", 1.0f, 1, 0xa5a5, Seq(Word("baz") -> 0.5f))), (c) => c.matchCount)
      result.label shouldBe "foo"
      result.probability shouldBe 0.75f
      result.matchCount shouldBe 2
      result.flags shouldBe 0xffff
      result.significantFeatures shouldBe Seq(Word("bar") -> 0.5f, Word("baz") -> 0.5f)
    }
  }

  describe("#reduce") {
    it("should only consider FORCED values if present") {
      val result = Classification.reduce(Seq(
        Classification("foo", 1.0f, 2, PredictResultFlag.mask(PredictResultFlag.FORCED), Seq(Word("bar") -> 1.0f)),
        Classification("foo", 0.0f, 6, PredictResultFlag.mask(PredictResultFlag.FORCED), Seq(Word("baz") -> 0.0f)),
        Classification("foo", 0.99f, 2, 0, Seq(Word("hello") -> 0.9f, Word("world") -> 0.9f))))
      result.label shouldBe "foo"
      result.probability shouldBe 0.25f
      result.matchCount shouldBe 8
      result.isForced shouldBe true
      result.flags shouldBe PredictResultFlag.mask(PredictResultFlag.FORCED)
      result.significantFeatures shouldBe Seq(Word("bar") -> 1.0f, Word("baz") -> 0.0f)
    }
  }
}
