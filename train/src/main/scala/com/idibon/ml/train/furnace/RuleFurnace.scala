package com.idibon.ml.train.furnace

import com.idibon.ml.common.Engine
import com.idibon.ml.predict.ensemble.ClassificationEnsembleModel
import com.idibon.ml.predict.rules.DocumentRules
import com.idibon.ml.predict.{PredictResult, Classification, PredictModel}
import com.idibon.ml.train.{Rule, TrainOptions}
import org.json4s._

/**
  * Base class for creating rule models.
  *
  * @param name the name to give the final model.
  * @tparam T whether we're dealing with Spans or (document) Classifications.
  */
abstract class RuleFurnace[+T <: PredictResult](name: String) {

  /** Trains the model synchronously
    *
    * @param options training data and options
    */
  def doHeat(options: TrainOptions): PredictModel[T] = {
    // create map of label -> list of rules
    val ruleMap = createLabelToRulesMap(options.rules)
    // create uuid label to human lable map
    val labelToName = options.labels.map(l => l.uuid.toString -> l.name).toMap
    // create rules
    createEnsembleRuleModel(ruleMap, labelToName)
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

  /**
    * Creates the rule ensemble model.
    *
    * @param ruleMap uuid label -> sequence of rules for that uuid label
    * @param labelToName uuid label -> human readable label
    * @return
    */
  def createEnsembleRuleModel(ruleMap: Map[String, Seq[(String, Float)]],
                              labelToName: Map[String, String]): PredictModel[T]
}

/**
  * Furnace for creating rules models.
  *
  * So that we only have to specify a single furnace in the configuration, and
  * without reference to specific labels, this furnace creates classification rules
  * models for all labels.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/24/16.
  */
class ClassificationRuleFurnace(val name: String) extends RuleFurnace[Classification](name)
  with Furnace2[Classification] {

  /**
    * Helper method to create span rule models from a map of rules.
    *
    * @param ruleMap uuid label -> sequence of rules for that uuid label
    * @param labelToName uuid label -> human readable label
    * @return uuid label -> SpanRule model
    */
  def createRuleModels(ruleMap: Map[String, Seq[(String, Float)]],
                       labelToName: Map[String, String]):
  Map[String, DocumentRules] = {
    ruleMap.map({ case (label, labelRules) =>
      label -> new DocumentRules(label, labelRules.toList)
    })
  }

  /**
    * Creates the classification rule ensemble model.
    *
    * @param ruleMap uuid label -> sequence of rules for that uuid label
    * @param labelToName uuid label -> human readable label
    * @return
    */
  override def createEnsembleRuleModel(ruleMap: Map[String, Seq[(String, Float)]],
                                       labelToName: Map[String, String]): PredictModel[Classification] = {
    val models: Map[String, PredictModel[Classification]] = createRuleModels(ruleMap, labelToName)
    // create span ensemble model to house it all
    //TODO: create training summaries
    new ClassificationEnsembleModel(name, models)
  }
}

object ClassificationRuleFurnace extends ((Engine, String, JObject) => Furnace2[Classification]) {

  /** Constructs a Classification Rule furnace for document scope rules.
    *
    * @param name furnace / model name
    * @param engine current IdiML engine content
    * @param json JSON configuration data for the Furnace
    */
  def apply(engine: Engine, name: String, json: JObject): ClassificationRuleFurnace = {
    new ClassificationRuleFurnace(name)
  }
}
