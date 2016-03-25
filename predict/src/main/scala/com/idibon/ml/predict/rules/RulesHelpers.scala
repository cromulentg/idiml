package com.idibon.ml.predict.rules

import java.util.regex.{Pattern}

import com.typesafe.scalalogging.Logger

import scala.util.{Try}

/**
  * Private static class for storing rule helper methods.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/24/16.
  */
private[rules] object RulesHelpers {

  /** Tests if the user-provided rule weight is valid
    *
    * @param w - weight
    * @return true if the weight is in the valid range, false otherwise
    */
  def isValidWeight(w: Float) = w >= 0.0 && w <= 1.0

  /**
    * Helper method to answer the question, whether the rule is a regular expression or not.
    *
    * @param rule
    * @return
    */
  def isRegexRule(rule: String): Boolean = {
    rule != null && rule.startsWith("/") && rule.endsWith("/") && rule.length() > 2
  }

  /** Tries to precompile rule phrases to Pattern instances
    *
    * @param rules - rule phrases to compile
    * @return A map from the raw rule phrase to the compilation results
    */
  def compileRules(rules: Iterable[String]) = {
    rules.par.map(phrase => {
      val pattern = Try({
        if (isRegexRule(phrase))
          Pattern.compile(phrase.substring(1, phrase.length() - 1))
        else
          Pattern.compile(phrase, Pattern.LITERAL | Pattern.CASE_INSENSITIVE)
      })
      (phrase, pattern)
    }).toList
  }
}

/**
  * Trait for storing regular expression rules.
  *
  */
trait RuleStorage {
  /** Extending classes must have a field rules which contains the expression and weight **/
  val rules: List[(String, Float)]
  /** To be able to log things about the rules being stored, we need a logger **/
  def getLogger: Logger

  // log a warning message if any of the rules has an invalid weight
  rules.filter(r => !RulesHelpers.isValidWeight(r._2))
    .foreach({ case (phrase, weight) => {
      getLogger.warn(s"[$this] ignoring invalid weight $weight for $phrase")
    }})
  /** Contains mapping of expression -> valid weight **/
  val ruleWeightMap = rules.filter(r => RulesHelpers.isValidWeight(r._2)).toMap
  /** List of expression & compiled expression. We iterate over this each time to find matches. **/
  val rulesCache = RulesHelpers.compileRules(ruleWeightMap.map(_._1))

  // log a warning message if any of the rules fail to compile
  rulesCache.filter(_._2.isFailure).foreach({ case (p, r) => {
    getLogger.warn(s"[$this] unable to compile $p: ${r.failed.get.getMessage}")
  }})
}
