package com.idibon.ml.predict.ensemble

import com.idibon.ml.alloy.IntentAlloy
import com.idibon.ml.predict.rules.DocumentRules
import com.idibon.ml.predict.{Document, PredictOptionsBuilder, SingleLabelDocumentResult}
import com.idibon.ml.common.EmbeddedEngine
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
      val docRules2 = new DocumentRules("alabel", List(("is", 0.5f)))
      val ensemble = new EnsembleModel("alabel", List(docRules1, docRules2))
      val metadata = ensemble.save(alloy.writer())
      val expectedMetadata = Some(JObject(List(
        ("label",JString("alabel")),
        ("size",JInt(2)),
        ("model-meta",JObject(List(
          ("0", JObject(List(
            ("config", JObject(List(("label",JString("alabel"))))),
            ("class",JString("com.idibon.ml.predict.rules.DocumentRules"))))),
          ("1",JObject(List(
            ("config", JObject(List(("label",JString("alabel"))))),
            ("class",JString("com.idibon.ml.predict.rules.DocumentRules"))))
            )))))))
      metadata shouldBe expectedMetadata
      val ensemble2 = (new EnsembleModelLoader).load(new EmbeddedEngine, alloy.reader(), metadata)
      ensemble shouldBe ensemble2
    }
//
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
          ("0", JObject(List(
            ("config", JObject(List(("label",JString("alabel"))))),
            ("class",JString("com.idibon.ml.predict.rules.DocumentRules"))))),
          ("1",JObject(List(
            ("config", JObject(List(("label",JString("alabel"))))),
            ("class",JString("com.idibon.ml.predict.rules.DocumentRules"))))
            )))))))
      metadata shouldBe expectedMetadata
      val ensemble2 = (new EnsembleModelLoader).load(new EmbeddedEngine, alloy.reader(), metadata)
      ensemble shouldBe ensemble2
    }
  }

  describe("document prediction test cases") {
    it("works as intended with one model") {
      val docRules = new DocumentRules("blabel", List(("/str[ij]ng/", 0.5f), ("is", 0.5f)))
      val ensembleModel = new EnsembleModel("blabel", List(docRules))
      val doc = new JObject(List("content" -> new JString("string matching is working")))
      val actual: SingleLabelDocumentResult = ensembleModel.predict(
        Document.document(doc), new PredictOptionsBuilder().build())
        .asInstanceOf[SingleLabelDocumentResult]
      actual.label shouldBe "blabel"
      actual.matchCount shouldBe 2
      actual.probability shouldEqual 0.5f
      actual.significantFeatures shouldEqual List()
    }

    it("works as intended with two models - taking the weighted average") {
      val docRules1 = new DocumentRules("blabel", List(("/str[ij]ng/", 0.5f), ("is", 0.5f)))
      val docRules2 = new DocumentRules("blabel", List(("/ma[th]ching/", 0.35f)))
      val ensembleModel = new EnsembleModel("blabel", List(docRules1, docRules2))
      val doc = new JObject(List("content" -> new JString("string matching is working")))
      val actual = ensembleModel.predict(
        Document.document(doc), new PredictOptionsBuilder()
          .showSignificantFeatures(0.0f)
          .build())
        .asInstanceOf[SingleLabelDocumentResult]
      actual.label shouldBe "blabel"
      actual.matchCount shouldBe 3
      actual.probability shouldEqual 0.45f +- 0.0001f // acount for floating point values
      actual.significantFeatures should contain theSameElementsAs List(("/str[ij]ng/",0.5f), ("is",0.5f), ("/ma[th]ching/",0.35f))
    }

    it("works as intended taking first model with black/white list trigger") {
      val docRules1 = new DocumentRules("blabel", List(("/str[ij]ng/", 0.0f), ("is", 1.0f)))
      val docRules2 = new DocumentRules("blabel", List(("/ma[th]ching/", 1.0f)))
      val ensembleModel = new EnsembleModel("blabel", List(docRules1, docRules2))
      val doc = new JObject(List("content" -> new JString("string matching is working")))
      val actual = ensembleModel.predict(
        Document.document(doc), new PredictOptionsBuilder()
          .showSignificantFeatures(0.0f)
          .build())
        .asInstanceOf[SingleLabelDocumentResult]
      actual.label shouldBe "blabel"
      actual.matchCount shouldBe 2
      actual.probability shouldEqual 0.5f
      actual.significantFeatures should contain theSameElementsAs List(("/str[ij]ng/", 0.0f), ("is", 1.0f))
    }

  }

  describe("vector prediction test cases") {
    it("works as intended") {
      //TODO: once I have a model that properly implements this
    }
  }
}
