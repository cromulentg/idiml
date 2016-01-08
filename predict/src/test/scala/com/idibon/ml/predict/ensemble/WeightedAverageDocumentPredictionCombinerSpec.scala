package com.idibon.ml.predict.ensemble

import com.idibon.ml.predict.{SingleLabelDocumentResultBuilder, PredictResult, SingleLabelDocumentResult}
import org.scalatest.{BeforeAndAfter, Matchers, FunSpec}

/**
  * Class to test the weighted average document prediction combiner.
  */
class WeightedAverageDocumentPredictionCombinerSpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("tests the combine function") {
    it("takes the first black or white list result") {
      val result1 = new SingleLabelDocumentResult(
        "string1", 2, 1.0, List(("feature1", 1.0)), 1,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> true))
      val result2 = new SingleLabelDocumentResult(
        "string2", 2, 1.0, List(("feature2", 1.0)), 2,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> true))
      val result3 = new SingleLabelDocumentResult(
        "string3", 2, 0.0, List(("feature3", 0.0)), 1,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> true))

      val wadpc = new WeightedAverageDocumentPredictionCombiner("mod1", 2)
      val actual = wadpc.combine(List(result1, result2, result3))
      val expected = new SingleLabelDocumentResultBuilder("mod1", 2)
        .copyFromExistingSingleLabelDocumentResult(result1).build()
      actual shouldBe expected
    }

    it("computes a proper weighted sum") {
      val result1 = new SingleLabelDocumentResult(
        "string1", 2, 0.6, List(("feature1", 0.6)), 1,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))
      val result2 = new SingleLabelDocumentResult(
        "string2", 2, 0.5, List(("feature2", 0.5)), 2,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))
      val result3 = new SingleLabelDocumentResult(
        "string3", 2, 0.4, List(("feature3", 0.4)), 1,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))

      val wadpc = new WeightedAverageDocumentPredictionCombiner("mod1", 2)
      val actual = wadpc.combine(List(result1, result2, result3))
      val expected = new SingleLabelDocumentResultBuilder("mod1", 2)
        .setProbability(0.5)
        .addSignificantFeatures(List(("feature1", 0.6), ("feature2", 0.5), ("feature3", 0.4)))
        .setMatchCount(4.0)
        .build()
      actual shouldBe expected
    }

    it("returns 0.0 if there are no matches") {
      val result1 = new SingleLabelDocumentResult(
        "string1", 2, 0.0, List(), 0,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))
      val result2 = new SingleLabelDocumentResult(
        "string2", 2, 0.0, List(), 0,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))
      val result3 = new SingleLabelDocumentResult(
        "string3", 2, 0.0, List(), 0,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))

      val wadpc = new WeightedAverageDocumentPredictionCombiner("mod1", 2)
      val actual = wadpc.combine(List(result1, result2, result3))
      val expected = new SingleLabelDocumentResultBuilder("mod1", 2)
        .setProbability(0.0)
        .addSignificantFeatures(List())
        .setMatchCount(0.0)
        .build()
      actual shouldBe expected
    }

    it("should only compute using values for the label initialized with") {
      val result1 = new SingleLabelDocumentResult(
        "string1", 1, 0.6, List(("feature1", 0.6)), 1,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))
      val result2 = new SingleLabelDocumentResult(
        "string2", 2, 0.5, List(("feature2", 0.5)), 2,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))
      val result3 = new SingleLabelDocumentResult(
        "string3", 3, 0.4, List(("feature3", 0.4)), 1,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))

      val wadpc = new WeightedAverageDocumentPredictionCombiner("mod1", 2)
      val actual = wadpc.combine(List(result1, result2, result3))
      val expected = new SingleLabelDocumentResultBuilder("mod1", 2)
        .setProbability(0.5)
        .addSignificantFeatures(List(("feature2", 0.5)))
        .setMatchCount(2.0)
        .build()
      actual shouldBe expected
    }

  }

}
