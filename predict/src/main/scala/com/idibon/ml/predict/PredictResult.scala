package com.idibon.ml.predict

import scala.collection.mutable

/**
  * Parent class of all prediction results.
  *
  * Assumptions:
  *   - Instantiations of this class would be case classes, since case classes are nice.
  *   - PredictResult is basically a placeholder til we feel out single label vs multiple label and
  *   the scope of what models/objects deal with the results. Since we might just cleave off worrying
  *   what high level object type we're dealing with, and instead just use the lower level case class
  *   throughout.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>"
  *
  * @param modelIdentifier a string representing a path to a model.
  * @param topLabel
  * @param topLabelProbability
  * @param topLabelSignificantFeatures
  * @param topLabelMatchCount
  * @param topLabelFlags
  */
abstract class PredictResult(modelIdentifier: String,
                             topLabel: String,
                             topLabelProbability: Double,
                             topLabelSignificantFeatures: List[(String, Double)],
                             topLabelMatchCount: Double,
                             topLabelFlags: Map[String, Boolean]) {
}

object PredictResult {
  // flags for special cases.
  val WHITELIST_OR_BLACKLIST = "WhitelistOrBlacklist"
}

/**
  * Class representing a single label prediction for a document.
  * @param modelIdentifier
  * @param label
  * @param probability
  * @param significantFeatures
  * @param matchCount
  * @param flags
  */
case class SingleLabelDocumentResult(modelIdentifier: String,
                                     label: String,
                                     probability: Double,
                                     significantFeatures: List[(String, Double)],
                                     matchCount: Double,
                                     flags: Map[String, Boolean]) extends PredictResult(
  modelIdentifier, label, probability, significantFeatures, matchCount, flags
)

/**
  * Class to build a single label document prediction result.
  * @param modelIdentifier a string representing a path to a model.
  * @param label
  */
class SingleLabelDocumentResultBuilder(modelIdentifier: String, label: String) {
  private var probability: Double = 0.0
  private val significantFeatures = scala.collection.mutable.MutableList[(String, Double)]()
  private var matchCount: Double = 0.0
  private val specialFlags = mutable.HashMap[String, Boolean]()

  /**
    * Sets the probability.
    * @param probability
    * @return
    */
  def setProbability(probability: Double): SingleLabelDocumentResultBuilder = {
    this.probability = probability
    this
  }

  /**
    * Adds to the current match count.
    * @param matchCount
    * @return
    */
  def addToMatchCount(matchCount: Double): SingleLabelDocumentResultBuilder = {
    this.matchCount += matchCount
    this
  }

  /**
    * Sets the match count.
    * @param matchCount
    * @return
    */
  def setMatchCount(matchCount: Double): SingleLabelDocumentResultBuilder = {
    this.matchCount = matchCount
    this
  }

  /**
    * Appends a feature to this result.
    * @param feature
    * @return
    */
  def addSignificantFeature(feature: (String, Double)): SingleLabelDocumentResultBuilder = {
    this.significantFeatures += feature
    this
  }

  /**
    * Adds a whole list to this result.
    * @param significantFeatures
    * @return
    */
  def addSignificantFeatures(significantFeatures: List[(String, Double)]): SingleLabelDocumentResultBuilder = {
    this.significantFeatures ++= significantFeatures
    this
  }

  /**
    * Sets flags that we use for decision logic.
    * @param name
    * @param value
    * @return
    */
  def setFlags(name: String, value: Boolean): SingleLabelDocumentResultBuilder = {
    specialFlags.put(name, value)
    this
  }

  /**
    * Copies from a previous single label result.
    * @param result
    * @return
    */
  def copyFromExistingSingleLabelDocumentResult(result: SingleLabelDocumentResult): SingleLabelDocumentResultBuilder = {
    this.probability = result.probability
    this.significantFeatures ++= result.significantFeatures
    this.matchCount = result.matchCount
    result.flags.foreach(x => specialFlags.put(x._1, x._2))
    this
  }

  /**
    * Builds the SingleLabelDocumentResult object.
    * @return
    */
  def build(): SingleLabelDocumentResult = {
    new SingleLabelDocumentResult(
      modelIdentifier, label, probability, significantFeatures.toList, matchCount, specialFlags.toMap)
  }

}
