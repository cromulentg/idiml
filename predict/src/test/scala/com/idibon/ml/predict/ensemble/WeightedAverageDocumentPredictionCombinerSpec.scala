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
        "string1", "a-label", 1.0f, List(("feature1", 1.0f)), 1,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> true))
      val result2 = new SingleLabelDocumentResult(
        "string2", "a-label", 1.0f, List(("feature2", 1.0f)), 2,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> true))
      val result3 = new SingleLabelDocumentResult(
        "string3", "a-label", 0.0f, List(("feature3", 0.0f)), 1,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> true))

      val wadpc = new WeightedAverageDocumentPredictionCombiner("mod1", "a-label")
      val actual = wadpc.combine(List(result1, result2, result3))
      val expected = new SingleLabelDocumentResultBuilder("mod1", "a-label")
        .copyFromExistingSingleLabelDocumentResult(result1).build()
      actual shouldBe expected
    }

    it("computes a proper weighted sum") {
      val result1 = new SingleLabelDocumentResult(
        "string1", "a-label", 0.6f, List(("feature1", 0.6f)), 1,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))
      val result2 = new SingleLabelDocumentResult(
        "string2", "a-label", 0.5f, List(("feature2", 0.5f)), 2,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))
      val result3 = new SingleLabelDocumentResult(
        "string3", "a-label", 0.4f, List(("feature3", 0.4f)), 1,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))

      val wadpc = new WeightedAverageDocumentPredictionCombiner("mod1", "a-label")
      val actual = wadpc.combine(List(result1, result2, result3))
      val expected = new SingleLabelDocumentResultBuilder("mod1", "a-label")
        .setProbability(0.5f)
        .addSignificantFeatures(List(("feature1", 0.6f), ("feature2", 0.5f), ("feature3", 0.4f)))
        .setMatchCount(4)
        .build()
      actual shouldBe expected
    }

    it("returns 0.0 if there are no matches") {
      val result1 = new SingleLabelDocumentResult(
        "string1", "a-label", 0.0f, List(), 0,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))
      val result2 = new SingleLabelDocumentResult(
        "string2", "a-label", 0.0f, List(), 0,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))
      val result3 = new SingleLabelDocumentResult(
        "string3", "a-label", 0.0f, List(), 0,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))

      val wadpc = new WeightedAverageDocumentPredictionCombiner("mod1", "a-label")
      val actual = wadpc.combine(List(result1, result2, result3))
      val expected = new SingleLabelDocumentResultBuilder("mod1", "a-label")
        .setProbability(0.0f)
        .addSignificantFeatures(List())
        .setMatchCount(0)
        .build()
      actual shouldBe expected
    }

    it("should only compute using values for the label initialized with") {
      val result1 = new SingleLabelDocumentResult(
        "string1", "x-label", 0.6f, List(("feature1", 0.6f)), 1,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))
      val result2 = new SingleLabelDocumentResult(
        "string2", "a-label", 0.5f, List(("feature2", 0.5f)), 2,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))
      val result3 = new SingleLabelDocumentResult(
        "string3", "z-label", 0.4f, List(("feature3", 0.4f)), 1,
        Map(PredictResult.WHITELIST_OR_BLACKLIST -> false))

      val wadpc = new WeightedAverageDocumentPredictionCombiner("mod1", "a-label")
      val actual = wadpc.combine(List(result1, result2, result3))
      val expected = new SingleLabelDocumentResultBuilder("mod1", "a-label")
        .setProbability(0.5f)
        .addSignificantFeatures(List(("feature2", 0.5f)))
        .setMatchCount(2)
        .build()
      actual shouldBe expected
    }

  }

}
