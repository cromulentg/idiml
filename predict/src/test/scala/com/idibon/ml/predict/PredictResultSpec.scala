package com.idibon.ml.predict

import com.idibon.ml.feature.bagofwords.Word
import com.idibon.ml.feature.{FeatureInputStream, FeatureOutputStream}

import java.io.{ByteArrayOutputStream, ByteArrayInputStream}
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

  describe("save and load") {
    it("should save and load correctly") {
      val s = Span("label", 0.75f, 0, 0, 5)
      val os = new ByteArrayOutputStream
      val fos = new FeatureOutputStream(os)
      fos.writeBuildable(s)
      fos.close
      val fis = new FeatureInputStream(new ByteArrayInputStream(os.toByteArray))
      fis.readBuildable.asInstanceOf[Span] shouldBe s
    }
  }

  describe("end point computation") {
    it("should compute the endpoint correctly") {
      val s = Span("label", 0.75f, 0, 2, 2)
      s.end shouldBe 4
    }
  }

  describe("choose between rule spans tests") {
    val chosen = Span("a-label", 0.6f, 2, 2, 2)
    it("handles chosen having better probability") {
      val contender = Span("b-label", 0.45f, 2, 2, 2)
      Span.chooseBetweenRuleSpans(chosen, contender) shouldBe chosen
    }
    it("handles contender having better probability"){
      val contender = Span("b-label", 0.65f, 2, 2, 2)
      Span.chooseBetweenRuleSpans(chosen, contender) shouldBe contender
    }
    it("handles tie breaking on length when chose is longer"){
      val contender = Span("b-label", 0.4f, 2, 2, 1)
      Span.chooseBetweenRuleSpans(chosen, contender) shouldBe chosen
    }
    it("handles tie breaking on length when contender is longer"){
      val contender = Span("b-label", 0.4f, 2, 2, 3)
      Span.chooseBetweenRuleSpans(chosen, contender) shouldBe contender
    }
    it("handles tie breaking on label name when chosen has lower label lexiograpically") {
      val contender = Span("b-label", 0.4f, 2, 2, 2)
      Span.chooseBetweenRuleSpans(chosen, contender) shouldBe chosen
    }
    it("handles tie breaking on label name when contender has lower label lexiograpically"){
      val contender = Span("a--label", 0.4f, 2, 2, 2)
      Span.chooseBetweenRuleSpans(chosen, contender) shouldBe contender
    }
  }
  describe("choose rules span greedily tests") {
    it("handles empty sequence") {
      val start = Span("a-label", 0.4f, 2, 2, 2)
      Span.chooseRuleSpanGreedily(start, Seq()) shouldBe start
    }
    it("handles single item list") {
      val start = Span("a-label", 0.4f, 2, 2, 2)
      val spans = Seq(Span("a-label", 0.5f, 2, 2, 2))
      Span.chooseRuleSpanGreedily(start, spans) shouldBe start
    }
    it("handles multi item list") {
      val start = Span("a-label", 0.4f, 2, 2, 2)
      val spans = Seq(Span("b-label", 0.6f, 2, 2, 3), Span("c-label", 0.6f, 2, 2, 4))
      Span.chooseRuleSpanGreedily(start, spans) shouldBe spans(1)
    }
  }
  describe("choose span tests") {
    it("handles empty spans") {
      val start = Span("a-label", 0.4f, 2, 2, 2)
      Span.chooseSpan(start, Seq()) shouldBe start
    }
    it("handles multiple rule spans") {
      // each is a different label and they overlap
      val start = Span("a-label", 0.4f, 2, 2, 2)
      val spans = Seq(Span("b-label", 0.6f, 2, 2, 3), Span("c-label", 0.6f, 2, 2, 4))
      Span.chooseSpan(start, spans) shouldBe spans(1)
    }
    it("handles a prediction span with a rule span") {
      // prediction as start
      val predictionStart = Span("a-label", 0.9f, 0, 2, 2)
      val spans = Seq(Span("b-label", 0.4f, 3, 2, 2))
      Span.chooseSpan(predictionStart, spans) shouldBe spans(0)
      // rule as start
      val ruleStart = Span("a-label", 0.2f, 2, 2, 2)
      val spansWithPrediction = Seq(Span("b-label", 0.9f, 0, 2, 2))
      Span.chooseSpan(ruleStart, spansWithPrediction) shouldBe ruleStart
    }
    it("handles rule, prediction, rule span overlap") {
      val ruleStart = Span("a-label", 0.2f, 2, 2, 2)
      val spansWithPrediction = Seq(Span("b-label", 0.9f, 0, 2, 2), Span("b-label", 0.3f, 3, 2, 5))
      Span.chooseSpan(ruleStart, spansWithPrediction) shouldBe ruleStart
    }
    it("throws exception when trying to merge to predict rules") {
      val ruleStart = Span("a-label", 0.2f, 0, 2, 2)
      val spansWithPrediction = Seq(Span("b-label", 0.9f, 0, 2, 2), Span("b-label", 0.3f, 0, 2, 5))
      intercept[IllegalStateException] {
        Span.chooseSpan(ruleStart, spansWithPrediction)
      }
    }
  }
  describe("get overlapping spans tests"){
    it("works with empty sequence") {
      Span.getOverlappingSpans(Span("a", 0.5f, 0, 2, 2), Seq()) shouldBe Seq()
    }
    it("works finding overlaps") {
      val spans = Seq(
        Span("b-label", 0.9f, 0, 2, 2),
        Span("z-label", 0.3f, 1, 2, 5),
        Span("c-label", 0.3f, 1, 8, 10))
      Span.getOverlappingSpans(Span("a", 0.5f, 0, 0, 3), spans) shouldBe spans.slice(0, 2)
    }
    it("works finding no overlaps") {
      /*
                10        20        30        40        50
      012345678901234567890123456789012345678901234567890
      A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
  "a" ---  offset is 0, end is 3
  "b-"   --  offset is 3, end is 5
  "z-"     ----- offset is 5, end is 10
  "c-"          ---------- offset is 10, end is 20
       */
      val spans = Seq(
        Span("b-label", 0.9f, 2, 3, 2),
        Span("z-label", 0.3f, 2, 5, 5),
        Span("c-label", 0.8f, 3, 10, 10))
      Span.getOverlappingSpans(Span("a", 0.5f, 0, 0, 3), spans) shouldBe Seq()
    }
  }
  describe("greedy reduce tests") {
    /** Dummy function for unit testing purposes **/
    def spanChooser(s: Span, spans: Seq[Span]): Span = {
      (s +: spans).minBy(z => z.label)
    }
    it("handles empty components") {
      Span.greedyReduce(Seq(), spanChooser) shouldBe Seq()
    }
    it("handles single item components") {
      val components = Seq(Span("a", 0.4f, 2, 2, 2))
      Span.greedyReduce(components, spanChooser) shouldBe components
    }
    it("handles no overlaps in components") {
      val components = Seq(
        Span("b-label", 0.9f, 2, 3, 2),
        Span("z-label", 0.3f, 2, 5, 5),
        Span("c-label", 0.8f, 3, 10, 10))
      Span.greedyReduce(components, spanChooser) shouldBe components
    }
    it("handles an overlap in components") {
      val components = Seq(
        Span("a", 0.5f, 0, 0, 3),
        Span("b-label", 0.9f, 2, 2, 2),
        Span("z-label", 0.3f, 2, 5, 5),
        Span("c-label", 0.8f, 3, 10, 10))
      Span.greedyReduce(components, spanChooser) shouldBe (components(0) +: components.slice(2,4))
    }
    it("handles multiple overlaps in components"){
      val components = Seq(
        Span("a", 0.5f, 0, 0, 3),
        Span("b-label", 0.9f, 2, 2, 2),
        Span("z-label", 0.3f, 2, 5, 6),
        Span("c-label", 0.8f, 3, 10, 10),
        Span("d-label", 0.8f, 3, 12, 4)
      )
      Span.greedyReduce(components, spanChooser) shouldBe Seq(components(0), components(3))
    }
  }
  describe("unionAndAverageOverlaps tests") {
    it("handles empty sequence") {
      Span.unionAndAverageOverlaps(Seq()) shouldBe Seq()
    }
    it("handles single item sequence") {
      val spans = Seq(Span("a", 0.4f, 0, 2, 2))
      Span.unionAndAverageOverlaps(spans) shouldBe spans
    }
    it("handles no overlaps") {
      val spans = Seq(Span("c-label", 0.9f, 2, 3, 2),
        Span("c-label", 0.6f, 2, 5, 5),
        Span("c-label", 0.8f, 3, 10, 10))
      Span.unionAndAverageOverlaps(spans) shouldBe spans
    }
    it("handles single set of overlapping spans") {
      val spans = Seq(
        Span("a", 0.5f, 0, 0, 3),
        Span("a", 0.8f, 2, 2, 4),
        Span("a", 0.6f, 2, 5, 6),
        Span("a", 0.7f, 3, 10, 10),
        Span("a", 0.9f, 3, 12, 4)
      )
      Span.unionAndAverageOverlaps(spans) shouldBe Seq(Span("a", 0.7f, 3, 0, 20))
    }
    it("handles multiple sets of overlapping spans") {
      val spans = Seq(
        Span("a", 0.7f, 0, 0, 3),
        Span("a", 0.8f, 2, 2, 2),
        Span("a", 0.6f, 2, 5, 5),
        Span("a", 0.8f, 3, 10, 10),
        Span("a", 0.9f, 3, 12, 4)
      )
      Span.unionAndAverageOverlaps(spans) shouldBe Seq(
        Span("a", 0.75f, 2, 0, 4),
        Span("a", 0.6f, 2, 5, 5),
        Span("a", 0.85f, 3, 10, 10))
    }
  }
  describe("getContiguousOverlappingSpans tests"){
    it("handles empty sequence") {
      Span.getContiguousOverlappingSpans(Span("A", 0.5f, 2, 2, 2), Seq()) shouldBe Seq()
    }
    it("handles no contiguous overlap") {
      val spans = Seq(Span("c-label", 0.9f, 2, 3, 2),
        Span("c-label", 0.6f, 2, 5, 5),
        Span("c-label", 0.8f, 3, 10, 10))
      val start = Span("c-label", 0.5f, 0, 0, 3)
      Span.getContiguousOverlappingSpans(start, spans) shouldBe Seq()
    }
    it("handles single overlap") {
      val spans = Seq(Span("c-label", 0.9f, 2, 1, 2),
        Span("c-label", 0.6f, 2, 5, 5),
        Span("c-label", 0.8f, 3, 10, 10))
      val start = Span("c-label", 0.5f, 0, 0, 3)
      Span.getContiguousOverlappingSpans(start, spans) shouldBe spans.slice(0, 1)
    }
    it("handles multiple contiguous overlap") {
      val spans = Seq(
        Span("a", 0.5f, 0, 1, 3),
        Span("a", 0.8f, 2, 2, 4),
        Span("a", 0.6f, 2, 5, 6),
        Span("a", 0.6f, 2, 6, 9),
        Span("a", 0.7f, 3, 10, 10),
        Span("a", 0.9f, 3, 12, 4)
      )
      val start = Span("a", 0.5f, 0, 0, 3)
      Span.getContiguousOverlappingSpans(start, spans) shouldBe spans
    }
    it("handles only getting overlapping with start span") {
      val spans = Seq(
        Span("a", 0.7f, 0, 1, 3),
        Span("a", 0.8f, 2, 2, 2),
        Span("a", 0.6f, 2, 5, 5),
        Span("a", 0.8f, 3, 10, 10),
        Span("a", 0.9f, 3, 12, 4)
      )
      val start = Span("a", 0.5f, 0, 0, 3)
      Span.getContiguousOverlappingSpans(start, spans) shouldBe spans.slice(0, 2)
    }
    it("handles tricky case to make sure current is assigned properly") {
      /* E.g. so we can handle the following
        1. [A, B, C, D]
        2.   [ B, C ]
        3.      [ C ]
        4.         [ D, E ]*/
      val spans = Seq(
        Span("a", 0.8f, 2, 1, 2),
        Span("a", 0.8f, 3, 2, 1),
        Span("a", 0.9f, 3, 3, 2)
      )
      val start = Span("a", 0.7f, 0, 0, 4)
      Span.getContiguousOverlappingSpans(start, spans) shouldBe spans
    }
  }
  describe("unionAndAverage tests"){
    it("throws assert exception on empty sequence") {
      intercept[AssertionError] {
        Span.unionAndAverage(Seq())
      }
    }
    it("handles single item sequence") {
      val span = Span("a", 0.5f, 0, 2, 2)
      Span.unionAndAverage(Seq(span)) shouldBe span
      val span2 = Span("a", 0.5f, 3, 2, 2)
      Span.unionAndAverage(Seq(span2)) shouldBe span2
    }
    it("handles multiple item sequence") {
      val spans = Seq(
        Span("a", 0.5f, 0, 0, 3),
        Span("a", 0.8f, 2, 2, 4),
        Span("a", 0.6f, 2, 5, 6),
        Span("a", 0.7f, 3, 10, 10),
        Span("a", 0.9f, 3, 12, 4)
      )
      Span.unionAndAverage(spans) shouldBe Span("a", 0.7f, 3, 0, 20)
    }
  }
}
