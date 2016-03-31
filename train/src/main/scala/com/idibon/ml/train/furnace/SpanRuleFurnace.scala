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
class SpanRuleFurnace(val name: String)
  extends Furnace2[Span] {

  /** Trains the model synchronously
    *
    * @param options training data and options
    */
  override protected def doHeat(options: TrainOptions): PredictModel[Span] = {
    // create map of label -> list of rules
    val ruleMap = createLabelToRulesMap(options.rules)
    // create uuid label to human lable map
    val labelToName = options.labels.map(l => l.uuid.toString -> l.name).toMap
    // create span rules
    val models: Map[String, SpanRules] = createSpanRuleModels(ruleMap, labelToName)
    // create span ensemble model to house it all
    //TODO: create training summaries
    new SpanEnsembleModel(name, models)
  }

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
    * Returns a map of label -> list of rules from the passed in rules list.
    *
    * @param rules
    * @return
    */
  def createLabelToRulesMap(rules: Seq[Rule]): Map[String, Seq[(String, Float)]] = {
    rules// get the bits we need out
      .map(r => (r.label, r.expression, r.weight))
      // group by label
      .groupBy({case (label, _, _) => label})
      // create label -> list of (expression, weight)
      .map({case (label, grouped) =>
      label -> grouped.map({case (l, e, w) => (e, w)})
    })
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
