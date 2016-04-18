package com.idibon.ml.train.furnace

import com.idibon.ml.common.Engine
import com.idibon.ml.predict.ensemble.SpanEnsembleModel
import com.idibon.ml.predict.rules.SpanRules
import com.idibon.ml.predict.{PredictModel, Span}
import com.idibon.ml.train.{Rule, TrainOptions}
import org.json4s._

/**
  * Furnace for creating rules models.
  *
  * So that we only have to specify a single furnace in the configuration, and
  * without reference to specific labels, this furnace creates span rules
  * models for all labels.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/24/16.
  */
class SpanRuleFurnace(val name: String) extends RuleFurnace[Span](name)
  with Furnace2[Span] {

  /**
    * Helper method to create span rule models from a map of rules.
    *
    * @param ruleMap uuid label -> sequence of rules for that uuid label
    * @param labelToName uuid label -> human readable label
    * @return uuid label -> SpanRule model
    */
  def createSpanRuleModels(ruleMap: Map[String, Seq[(String, Float)]],
                           labelToName: Map[String, String]):
  Map[String, SpanRules] = {
    ruleMap.map({ case (label, labelRules) =>
      label -> new SpanRules(label, labelToName(label), labelRules.toList)
    })
  }

  /**
    * Creates the span ensemble model.
    *
    * @param ruleMap uuid label -> sequence of rules for that uuid label
    * @param labelToName uuid label -> human readable label
    * @return
    */
  override def createEnsembleRuleModel(ruleMap: Map[String, Seq[(String, Float)]],
                                       labelToName: Map[String, String]): PredictModel[Span] = {
    val models: Map[String, SpanRules] = createSpanRuleModels(ruleMap, labelToName)
    // create span ensemble model to house it all
    //TODO: create training summaries
    new SpanEnsembleModel(name, models)
  }
}


object SpanRuleFurnace extends ((Engine, String, JObject) => Furnace2[Span]) {

  /** Constructs a ChainNERFurnace to train NER models
    *
    * @param name furnace / model name
    * @param engine current IdiML engine content
    * @param json JSON configuration data for the CRFFurnace
    */
  def apply(engine: Engine, name: String, json: JObject): SpanRuleFurnace = {
    new SpanRuleFurnace(name)
  }
}
