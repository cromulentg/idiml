package com.idibon.ml.predict

import java.io.{ByteArrayOutputStream, ByteArrayInputStream}

import com.idibon.ml.feature.bagofwords.Word
import com.idibon.ml.feature.{FeatureInputStream, FeatureOutputStream}

import org.scalatest.{Matchers, FunSpec}


/**
  * Tests the PredictResult trait functions.
  */
class PredictResultSpec extends FunSpec with Matchers {
  val pr1 = new PredictResult {
    override def probability: Float = 0.25f
    override def matchCount: Int = 3
    override def label: String = "monkeys"
    override def flags: Int = 123
  }

  describe("isCloseEnough tests"){
    it("matches on everything") {
      val prAlpha = new PredictResult {
        override def probability: Float = 0.25f
        override def matchCount: Int = 3
        override def label: String = "monkeys"
        override def flags: Int = 123
      }
      prAlpha.isCloseEnough(pr1) shouldBe true
    }
    it("matches within tolerance") {
      val prAlpha = new PredictResult {
        override def probability: Float = 0.25002f
        override def matchCount: Int = 3
        override def label: String = "monkeys"
        override def flags: Int = 123
      }
      prAlpha.isCloseEnough(pr1) shouldBe true
    }
    it("fails on label") {
      val prAlpha = new PredictResult {
        override def probability: Float = 0.25f
        override def matchCount: Int = 3
        override def label: String = "bananas"
        override def flags: Int = 123
      }
      prAlpha.isCloseEnough(pr1) shouldBe false
    }
    it("fails on matchCount") {
      val prAlpha = new PredictResult {
        override def probability: Float = 0.25f
        override def matchCount: Int = 234
        override def label: String = "monkeys"
        override def flags: Int = 123
      }
      prAlpha.isCloseEnough(pr1) shouldBe false
    }
    it("fails on flags") {
      val prAlpha = new PredictResult {
        override def probability: Float = 0.25f
        override def matchCount: Int = 3
        override def label: String = "monkeys"
        override def flags: Int = 12234
      }
      prAlpha.isCloseEnough(pr1) shouldBe false
    }
    it("fails on force flag") {
      val prAlpha = new PredictResult {
        override def probability: Float = 0.25f
        override def matchCount: Int = 3
        override def label: String = "monkeys"
        override def flags: Int = 2
      }
      prAlpha.isCloseEnough(pr1) shouldBe false
    }
    it("fails on rule flag") {
      val prAlpha = new PredictResult {
        override def probability: Float = 0.25f
        override def matchCount: Int = 3
        override def label: String = "monkeys"
        override def flags: Int = 1
      }
      prAlpha.isCloseEnough(pr1) shouldBe false
    }
    it("fails on float tolerance"){
      val prAlpha = new PredictResult {
        override def probability: Float = 0.252f
        override def matchCount: Int = 3
        override def label: String = "monkeys"
        override def flags: Int = 123
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

  it("should save and load correctly") {
    val c = Classification("foo", 0.875f, 1, 0, Seq(Word("foo") -> 0.5f))
    val os = new ByteArrayOutputStream
    val fos = new FeatureOutputStream(os)
    fos.writeBuildable(c)
    fos.close
    val fis = new FeatureInputStream(new ByteArrayInputStream(os.toByteArray))
    fis.readBuildable.asInstanceOf[Classification] shouldBe c
  }

  describe("#weighted_average") {
    it("should return a 0.0 confidence if there is no weight returned") {
      val result = Classification.weighted_average(Seq(
        Classification("foo", 1.0f, 1, 0, Seq()),
        Classification("foo", 1.0f, 1, 0, Seq())), (c) => 0)
      result.label shouldBe "foo"
      result.probability shouldBe 0.0f
      result.matchCount shouldBe 0
    }

    it("should perform a weighted average of probabilities") {
      val result = Classification.weighted_average(Seq(
        Classification("foo", 1.0f, 2, 0, Seq()),
        Classification("foo", 0.0f, 1, 0, Seq()),
        Classification("foo", 0.0f, 1, 0, Seq())), (c) => c.matchCount)
      result.label shouldBe "foo"
      result.probability shouldBe 0.5f
      result.matchCount shouldBe 4
      result.isForced shouldBe false
    }

    it("should return the superset of flags and significant features") {
      val result = Classification.weighted_average(Seq(
        Classification("foo", 0.5f, 1, 0x5a5a, Seq(Word("bar") -> 0.5f)),
        Classification("foo", 1.0f, 1, 0xa5a5, Seq(Word("baz") -> 0.5f))), (c) => c.matchCount)
      result.label shouldBe "foo"
      result.probability shouldBe 0.75f
      result.matchCount shouldBe 2
      result.flags shouldBe 0xffff
      result.significantFeatures shouldBe Seq(Word("bar") -> 0.5f, Word("baz") -> 0.5f)
    }
  }

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
        Classification("foo", 1.0f, 1, PredictResultFlag.mask(PredictResultFlag.RULE), Seq()),
        Classification("foo", 1.0f, 1, 0, Seq())), (c) => 0)
      result.label shouldBe "foo"
      result.probability shouldBe 0.0f
      result.matchCount shouldBe 0
    }

    it("should perform a weighted average of probabilities from one type") {
      val result = Classification.average(Seq(
        Classification("foo", 1.0f, 2, 0, Seq()),
        Classification("foo", 0.0f, 1, 0, Seq()),
        Classification("foo", 0.0f, 1, 0, Seq())), (c) => c.matchCount)
      result.label shouldBe "foo"
      result.probability shouldBe 0.5f
      result.matchCount shouldBe 4
      result.isForced shouldBe false
    }

    it("should perform the harmonic mean of probabilities from different types") {
      //this is an example case from tyler used to highlight the need for averaging model + rule results
      val result = Classification.average(Seq(
        Classification("tyler", 0.436334685364591f, 1, 0, Seq()),
        Classification("tyler", 0.99f, 1, PredictResultFlag.mask(PredictResultFlag.RULE), Seq()),
        Classification("tyler", 0.3828143710711002f, 1, 0, Seq()),
        Classification("tyler", 0.99f, 1, PredictResultFlag.mask(PredictResultFlag.RULE), Seq()),
        Classification("tyler", 0.41341606110226836f, 1, 0, Seq())), (c) => c.matchCount)
      result.label shouldBe "tyler"
      result.probability shouldBe 0.5807118f
      result.matchCount shouldBe 5
    }
  }

  describe("#reduce") {
    it("should only consider FORCED values if present") {
      val result = Classification.reduce(Seq(
        Classification("foo", 1.0f, 2, PredictResultFlag.mask(PredictResultFlag.FORCED,PredictResultFlag.RULE), Seq(Word("bar") -> 1.0f)),
        Classification("foo", 0.0f, 6, PredictResultFlag.mask(PredictResultFlag.FORCED,PredictResultFlag.RULE), Seq(Word("baz") -> 0.0f)),
        Classification("foo", 0.99f, 2, 0, Seq(Word("hello") -> 0.9f, Word("world") -> 0.9f))))
      result.label shouldBe "foo"
      result.probability shouldBe 0.25f
      result.matchCount shouldBe 8
      result.isForced shouldBe true
      result.flags shouldBe PredictResultFlag.mask(PredictResultFlag.FORCED,PredictResultFlag.RULE)
      result.significantFeatures shouldBe Seq(Word("bar") -> 1.0f, Word("baz") -> 0.0f)
    }
  }
}

class SpanSpec extends FunSpec with Matchers {
  it("should save and load correctly") {
    val s = Span("label", 0.75f, 0, 0, 5)
    val os = new ByteArrayOutputStream
    val fos = new FeatureOutputStream(os)
    fos.writeBuildable(s)
    fos.close
    val fis = new FeatureInputStream(new ByteArrayInputStream(os.toByteArray))
    fis.readBuildable.asInstanceOf[Span] shouldBe s
  }

  it("should compute the endpoint correctly") {
    val s = Span("label", 0.75f, 0, 2, 2)
    s.end shouldBe 4
  }
}
