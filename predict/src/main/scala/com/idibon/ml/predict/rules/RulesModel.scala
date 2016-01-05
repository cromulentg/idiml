package com.idibon.ml.predict.rules

import com.idibon.ml.predict.PredictModel

/**
  * Abstract parent class of all rule models.
  * @param label the index of the label these rules are for
  * @param rules a list of tuples of rule & weights.
  */
abstract class RulesModel(label: Int, rules: List[(String, Double)]) extends PredictModel {

}
