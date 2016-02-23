package com.idibon.ml.predict.rules

import com.idibon.ml.feature.Feature

/** Simple feature type used to represent a rule */
case class RuleFeature(phrase: String) extends Feature[String] {
  def get = phrase

  def getAsString: Option[String] = ???
}
