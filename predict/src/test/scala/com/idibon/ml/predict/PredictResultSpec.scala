package com.idibon.ml.predict

import org.scalatest.{Matchers, FunSpec}
import com.idibon.ml.feature.bagofwords.Word



/**
  * Tests the PredictResult trait functions.
  */
class PredictResultSpec extends FunSpec with Matchers {
  val pr1 = new PredictResult {
    override def probability: Float = 0.25f
    override def matchCount: Int = 3
    override def label: String = "monkeys"
    override def flags: Int = 123
    override def modelType:  PredictTypeFlag.Value = PredictTypeFlag.RULE
  }

  describe("isCloseEnough tests"){
    it("matches on everything") {
      val prAlpha = new PredictResult {
        override def probability: Float = 0.25f
        override def matchCount: Int = 3
        override def label: String = "monkeys"
        override def flags: Int = 123
        override def modelType:  PredictTypeFlag.Value = PredictTypeFlag.RULE
      }
      prAlpha.isCloseEnough(pr1) shouldBe true
    }
    it("matches within tolerance") {
      val prAlpha = new PredictResult {
        override def probability: Float = 0.25002f
        override def matchCount: Int = 3
        override def label: String = "monkeys"
        override def flags: Int = 123
        override def modelType:  PredictTypeFlag.Value = PredictTypeFlag.RULE
      }
      prAlpha.isCloseEnough(pr1) shouldBe true
    }
    it("fails on label") {
      val prAlpha = new PredictResult {
        override def probability: Float = 0.25f
        override def matchCount: Int = 3
        override def label: String = "bananas"
        override def flags: Int = 123
        override def modelType:  PredictTypeFlag.Value = PredictTypeFlag.RULE
      }
      prAlpha.isCloseEnough(pr1) shouldBe false
    }
    it("fails on matchCount") {
      val prAlpha = new PredictResult {
        override def probability: Float = 0.25f
        override def matchCount: Int = 234
        override def label: String = "monkeys"
        override def flags: Int = 123
        override def modelType:  PredictTypeFlag.Value = PredictTypeFlag.RULE
      }
      prAlpha.isCloseEnough(pr1) shouldBe false
    }
    it("fails on flags") {
      val prAlpha = new PredictResult {
        override def probability: Float = 0.25f
        override def matchCount: Int = 3
        override def label: String = "monkeys"
        override def flags: Int = 12234
        override def modelType:  PredictTypeFlag.Value = PredictTypeFlag.RULE
      }
      prAlpha.isCloseEnough(pr1) shouldBe false
    }
    it("fails on isForces") {
      val prAlpha = new PredictResult {
        override def probability: Float = 0.25f
        override def matchCount: Int = 3
        override def label: String = "monkeys"
        override def flags: Int = 123
        override def isForced = false
        override def modelType:  PredictTypeFlag.Value = PredictTypeFlag.RULE
      }
      prAlpha.isCloseEnough(pr1) shouldBe false
    }
    it("fails on float tolerance"){
      val prAlpha = new PredictResult {
        override def probability: Float = 0.252f
        override def matchCount: Int = 3
        override def label: String = "monkeys"
        override def flags: Int = 123
        override def modelType:  PredictTypeFlag.Value = PredictTypeFlag.RULE
      }
      prAlpha.isCloseEnough(pr1) shouldBe false
    }
  }

  describe("floatIsCloseEnough tests") {
    it("works on exact") {
      PredictResult.floatIsCloseEnough(0.001f, 0.001f) shouldBe true
    }
    it("works within tolerance") {
      PredictResult.floatIsCloseEnough(0.001f, 0.0015f) shouldBe true
    }
    it("fails outside of tolerance") {
      PredictResult.floatIsCloseEnough(0.001f, 0.000f) shouldBe false
    }
  }
}

class ClassificationSpec extends FunSpec with Matchers {

  describe("#average") {

    it("should return a 0.0 confidence if there is no weight returned") {
      val result = Classification.average(Seq(
        Classification("foo", 1.0f, 1, 0, Seq(), PredictTypeFlag.RULE),
        Classification("foo", 1.0f, 1, 0, Seq(), PredictTypeFlag.MODEL)), (c) => 0)
      result.label shouldBe "foo"
      result.probability shouldBe 0.0f
      result.matchCount shouldBe 0
    }

    it("should perform a weighted average of probabilities") {
      val result = Classification.average(Seq(
        Classification("foo", 1.0f, 2, 0, Seq(), PredictTypeFlag.RULE),
        Classification("foo", 0.0f, 1, 0, Seq(), PredictTypeFlag.MODEL),
        Classification("foo", 0.0f, 1, 0, Seq(), PredictTypeFlag.RULE)), (c) => c.matchCount)
      result.label shouldBe "foo"
      result.probability shouldBe 0.5f
      result.matchCount shouldBe 4
      result.isForced shouldBe false
    }

    it("should return the superset of flags and significant features") {
      val result = Classification.average(Seq(
        Classification("foo", 0.5f, 1, 0x5a5a, Seq(Word("bar") -> 0.5f), PredictTypeFlag.MODEL),
        Classification("foo", 1.0f, 1, 0xa5a5, Seq(Word("baz") -> 0.5f), PredictTypeFlag.RULE)), (c) => c.matchCount)
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
        Classification("foo", 1.0f, 2, PredictResultFlag.mask(PredictResultFlag.FORCED), Seq(Word("bar") -> 1.0f), PredictTypeFlag.RULE),
        Classification("foo", 0.0f, 6, PredictResultFlag.mask(PredictResultFlag.FORCED), Seq(Word("baz") -> 0.0f), PredictTypeFlag.RULE),
        Classification("foo", 0.99f, 2, 0, Seq(Word("hello") -> 0.9f, Word("world") -> 0.9f), PredictTypeFlag.RULE)))
      result.label shouldBe "foo"
      result.probability shouldBe 0.25f
      result.matchCount shouldBe 8
      result.isForced shouldBe true
      result.flags shouldBe PredictResultFlag.mask(PredictResultFlag.FORCED)
      result.significantFeatures shouldBe Seq(Word("bar") -> 1.0f, Word("baz") -> 0.0f)
    }
  }

  describe("#harmonic_mean") {
    it("should raise an exception if you average multiple labels") {
      intercept[IllegalArgumentException] {
        Classification.harmonic_mean(Seq(
          Classification("foo", 1.0f, 1, 0, Seq(), PredictTypeFlag.MODEL),
          Classification("bar", 1.0f, 1, 0, Seq(), PredictTypeFlag.RULE)), (c) => c.matchCount)
      }
    }

    it("should return a 0.0 confidence if there is no weight returned") {
      val result = Classification.harmonic_mean(Seq(
        Classification("foo", 1.0f, 1, 0, Seq(), PredictTypeFlag.RULE),
        Classification("foo", 1.0f, 1, 0, Seq(), PredictTypeFlag.MODEL)), (c) => 0)
      result.label shouldBe "foo"
      result.probability shouldBe 0.0f
      result.matchCount shouldBe 0
    }

    it("should perform a weighted average of probabilities from one type") {
      val result = Classification.harmonic_mean(Seq(
        Classification("foo", 1.0f, 2, 0, Seq(), PredictTypeFlag.RULE),
        Classification("foo", 0.0f, 1, 0, Seq(), PredictTypeFlag.RULE),
        Classification("foo", 0.0f, 1, 0, Seq(), PredictTypeFlag.RULE)), (c) => c.matchCount)
      result.label shouldBe "foo"
      result.probability shouldBe 0.5f
      result.matchCount shouldBe 4
      result.isForced shouldBe false
    }

    it("should perform the harmonic mean of probabilities from different types") {
      //this is an example case from tyler used to highlight the need for averaging model + rule results
      val result = Classification.harmonic_mean(Seq(
        Classification("tyler", 0.436334685364591f, 1, 0, Seq(), PredictTypeFlag.MODEL),
        Classification("tyler", 0.99f, 1, 0, Seq(), PredictTypeFlag.RULE),
        Classification("tyler", 0.3828143710711002f, 1, 0, Seq(), PredictTypeFlag.MODEL),
        Classification("tyler", 0.99f, 1, 0, Seq(), PredictTypeFlag.RULE),
        Classification("tyler", 0.41341606110226836f, 1, 0, Seq(), PredictTypeFlag.MODEL)), (c) => c.matchCount)

      result.label shouldBe "tyler"
      result.probability shouldBe 0.5807118f
      result.matchCount shouldBe 5
    }
  }
}