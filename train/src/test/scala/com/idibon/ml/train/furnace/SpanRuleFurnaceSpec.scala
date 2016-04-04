package com.idibon.ml.train.furnace

import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.predict._
import com.idibon.ml.predict.ensemble.{EnsembleModel, SpanEnsembleModel}
import com.idibon.ml.predict.rules.SpanRules
import com.idibon.ml.train.{TrainOptionsBuilder, TrainOptions, Rule}
import org.json4s.JsonAST.{JObject}
import org.json4s._
import org.scalatest.{FunSpec, Matchers}

class SpanRuleFurnaceSpec extends FunSpec with Matchers {


  val furnaceConfig = JObject(List(("config", JNull)))

  describe("apply & registry tests") {
    implicit val formats = org.json4s.DefaultFormats

    it("should raise an exception for invalid result types") {
      intercept[NoSuchElementException] {
        Furnace2[NotARealResult](null, "SpanRuleFurnace", "new furnace", null)
      }
      intercept[NoSuchElementException] {
        Furnace2[Classification](new EmbeddedEngine, "SpanRuleFurnace",
          "new furnace", (furnaceConfig \ "config").extract[JObject])
      }
    }

    it("should return the correct furnace") {
      val furnace = Furnace2[Span](new EmbeddedEngine, "SpanRuleFurnace",
        "new furnace", (furnaceConfig \ "config").extract[JObject])
      furnace.name shouldBe "new furnace"
      furnace shouldBe a [SpanRuleFurnace]
    }
  }

  describe("do heat tests") {
    it("handles no rules") {
      val sf = new SRFextender()
      val actual = sf.doHeat(new TrainOptionsBuilder().build(Seq()))
      actual.isInstanceOf[SpanEnsembleModel] shouldBe true
    }
    it("works as expected") {
      val labels = Seq(
        new Label("2a9fa8b0-e5e7-4126-a0a9-621a08fea145", "Negative"),
        new Label("bc4a112a-c68c-48ff-97cf-2fae4d65fe56", "Neutral"),
        new Label("491b3e5c-e749-4d61-8647-654d39acdd4b", "Positive")
      )
      val rules = Seq(
        ("2a9fa8b0-e5e7-4126-a0a9-621a08fea145", "is", 0.5f),
        ("2a9fa8b0-e5e7-4126-a0a9-621a08fea145", "a", 0.5f),
        ("2a9fa8b0-e5e7-4126-a0a9-621a08fea145", "monkey", 0.5f),
        ("bc4a112a-c68c-48ff-97cf-2fae4d65fe56", "banana", 0.5f)
      )
      val options = new TrainOptionsBuilder().addRules(rules).build(labels)
      val sf = SRFextender()
      val actual = sf.doHeat(options).asInstanceOf[SpanEnsembleModel]
      // do some reflection to get at the internal models
      val modelsField = classOf[SpanEnsembleModel].getDeclaredField("models")
      modelsField.setAccessible(true)
      val models = modelsField.get(actual).asInstanceOf[Map[String, PredictModel[Span]]]
      models.size shouldBe 2
      models("2a9fa8b0-e5e7-4126-a0a9-621a08fea145").isInstanceOf[SpanRules] shouldBe true
      models("bc4a112a-c68c-48ff-97cf-2fae4d65fe56").isInstanceOf[SpanRules] shouldBe true
    }
  }

  describe("create span rule models tests") {
    it("handles creating rule models corresponding to passed in rules") {
      val sf = new SpanRuleFurnace("test")
      val ruleMap = Map(
        "a" -> Seq(("is", 0.4f), ("monkey", 0.3f)),
        "c" -> Seq(("banana", 0.9f)))
      val labelToName = Map("a" -> "A", "b" -> "B", "c" -> "C")
      val actual = sf.createSpanRuleModels(ruleMap, labelToName)
      actual.size shouldBe 2
      actual("a") shouldBe new SpanRules("a", "A", ruleMap("a").toList)
      actual("c") shouldBe new SpanRules("c", "C", ruleMap("c").toList)
    }
    it("handles empty rule map"){
      val sf = new SpanRuleFurnace("test")
      val ruleMap = Map[String, Seq[(String, Float)]]()
      val labelToName = Map("a" -> "A", "b" -> "B", "c" -> "C")
      val actual = sf.createSpanRuleModels(ruleMap, labelToName)
      actual.size shouldBe 0
    }
    it("handles empty rules in rule map"){
      val sf = new SpanRuleFurnace("test")
      val ruleMap = Map(
        "a" -> Seq(("is", 0.4f), ("monkey", 0.3f)),
        "b" -> Seq(),
        "c" -> Seq(("banana", 0.9f)))
      val labelToName = Map("a" -> "A", "b" -> "B", "c" -> "C")
      val actual = sf.createSpanRuleModels(ruleMap, labelToName)
      actual.size shouldBe 3
      actual("a") shouldBe new SpanRules("a", "A", ruleMap("a").toList)
      actual("b") shouldBe new SpanRules("b", "B", List())
      actual("c") shouldBe new SpanRules("c", "C", ruleMap("c").toList)
    }
  }

  describe("createLabelToRulesMap tests") {
    val sf = new SpanRuleFurnace("test")
    it("handles empty rules") {
      sf.createLabelToRulesMap(Seq()) shouldBe Map()
    }
    it("handles single label rules") {
      val rules = Seq(
        new Rule("a", "is", 0.5f),
        new Rule("a", "a", 0.5f),
        new Rule("a", "monkey", 0.5f)
      )
      sf.createLabelToRulesMap(rules) shouldBe Map("a" ->
        Seq(("is", 0.5f), ("a", 0.5f), ("monkey", 0.5f)))
    }
    it("handles multi label rules") {
      val rules = Seq(
        new Rule("a", "is", 0.5f),
        new Rule("b", "a", 0.5f),
        new Rule("c", "monkey", 0.5f)
      )
      sf.createLabelToRulesMap(rules) shouldBe Map("a" -> Seq(("is", 0.5f)),
        "b" -> Seq(("a", 0.5f)), "c" -> Seq(("monkey", 0.5f)))
    }
  }
}

/** Class to make a protected method public for unit testing. **/
case class SRFextender() extends SpanRuleFurnace("test") {

  override def doHeat(options: TrainOptions): PredictModel[Span] = {
    super.doHeat(options)
  }
}

