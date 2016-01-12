package com.idibon.ml.predict.ensemble

import com.idibon.ml.alloy.IntentAlloy
import com.idibon.ml.predict.{PredictOptionsBuilder, SingleLabelDocumentResult, PredictResult}
import com.idibon.ml.predict.rules.DocumentRules
import org.json4s._
import org.scalatest.{BeforeAndAfter, FunSpec, Matchers}

/**
  * Class to test the Ensemble Model
  */
class EnsembleModelSpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("save and load") {
    it("should save and load properly") {
      val alloy = new IntentAlloy()
      val docRules1 = new DocumentRules("alabel", List())
      val docRules2 = new DocumentRules("alabel", List(("is", 0.5)))
      val ensemble = new EnsembleModel("alabel", List(docRules1, docRules2))
      val metadata = ensemble.save(alloy.writer())
      val expectedMetadata = Some(JObject(List(
        ("label",JString("alabel")),
        ("size",JInt(2)),
        ("model-meta",JObject(List(
          ("0",JObject(List(("label",JString("alabel"))))),
          ("1",JObject(List(("label",JString("alabel")))))))),
        ("model-index",JObject(List(
          ("0",JString("com.idibon.ml.predict.rules.DocumentRules")),
          ("1",JString("com.idibon.ml.predict.rules.DocumentRules"))))))))
      metadata shouldBe expectedMetadata
      val ensemble2 = new EnsembleModel()
      ensemble2.load(alloy.reader(), metadata)
      ensemble shouldBe ensemble2
    }

    it("should save and load empty models") {
      val alloy = new IntentAlloy()
      val docRules1 = new DocumentRules("alabel", List())
      val docRules2 = new DocumentRules("alabel", List())
      val ensemble = new EnsembleModel("alabel", List(docRules1, docRules2))
      val metadata = ensemble.save(alloy.writer())
      val expectedMetadata = Some(JObject(List(
        ("label",JString("alabel")),
        ("size",JInt(2)),
        ("model-meta",JObject(List(
          ("0",JObject(List(("label",JString("alabel"))))),
          ("1",JObject(List(("label",JString("alabel")))))))),
        ("model-index",JObject(List(
          ("0",JString("com.idibon.ml.predict.rules.DocumentRules")),
          ("1",JString("com.idibon.ml.predict.rules.DocumentRules"))))))))
      metadata shouldBe expectedMetadata
      val ensemble2 = new EnsembleModel()
      ensemble2.load(alloy.reader(), metadata)
      ensemble shouldBe ensemble2
    }
  }

  describe("document prediction test cases") {
    it("works as intended with one model") {
      val docRules = new DocumentRules("blabel", List(("/str[ij]ng/", 0.5), ("is", 0.5)))
      val ensembleModel = new EnsembleModel("blabel", List(docRules))
      val doc = new JObject(List("content" -> new JString("string matching is working")))
      val actual: SingleLabelDocumentResult = ensembleModel.predict(
        doc, new PredictOptionsBuilder().build())
        .asInstanceOf[SingleLabelDocumentResult]
      actual.label shouldBe "blabel"
      actual.matchCount shouldBe 2.0
      actual.probability shouldEqual 0.5
      actual.significantFeatures shouldEqual List()
    }

    it("works as intended with two models - taking the weighted average") {
      val docRules1 = new DocumentRules("blabel", List(("/str[ij]ng/", 0.5), ("is", 0.5)))
      val docRules2 = new DocumentRules("blabel", List(("/ma[th]ching/", 0.35)))
      val ensembleModel = new EnsembleModel("blabel", List(docRules1, docRules2))
      val doc = new JObject(List("content" -> new JString("string matching is working")))
      val actual = ensembleModel.predict(
        doc, new PredictOptionsBuilder()
          .showSignificantFeatures(0.0)
          .build())
        .asInstanceOf[SingleLabelDocumentResult]
      actual.label shouldBe "blabel"
      actual.matchCount shouldBe 3.0
      actual.probability shouldEqual 0.45
      actual.significantFeatures shouldEqual List(("/str[ij]ng/",0.5), ("is",0.5), ("/ma[th]ching/",0.35))
    }

    it("works as intended taking first model with black/white list trigger") {
      val docRules1 = new DocumentRules("blabel", List(("/str[ij]ng/", 0.0), ("is", 1.0)))
      val docRules2 = new DocumentRules("blabel", List(("/ma[th]ching/", 1.0)))
      val ensembleModel = new EnsembleModel("blabel", List(docRules1, docRules2))
      val doc = new JObject(List("content" -> new JString("string matching is working")))
      val actual = ensembleModel.predict(
        doc, new PredictOptionsBuilder()
          .showSignificantFeatures(0.0)
          .build())
        .asInstanceOf[SingleLabelDocumentResult]
      actual.label shouldBe "blabel"
      actual.matchCount shouldBe 2.0
      actual.probability shouldEqual 0.5
      actual.significantFeatures shouldEqual List(("/str[ij]ng/", 0.0), ("is", 1.0))
    }

  }

  describe("vector prediction test cases") {
    it("works as intended") {
      //TODO: once I have a model that properly implements this
    }
  }
}
