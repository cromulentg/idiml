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
  * @param modelIdentifier a string representing a model, right now the class name.
  * @param topLabel
  * @param topLabelProbability
  * @param topLabelSignificantFeatures
  * @param topLabelMatchCount
  * @param topLabelFlags
  */
abstract class PredictResult(modelIdentifier: String,
                             topLabel: String,
                             topLabelProbability: Float,
                             topLabelSignificantFeatures: List[(String, Float)],
                             topLabelMatchCount: Int,
                             topLabelFlags: Map[PredictResultFlag.Value, Boolean]) {
}

/**
  * Flags for special cases that we need to be aware of during prediction.
  */
object PredictResultFlag extends Enumeration {
  type PredictResultFlag = Value
  val FORCED // e.g. used when a blacklist or whitelist rule is used.
    = Value
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
                                     probability: Float,
                                     significantFeatures: List[(String, Float)],
                                     matchCount: Int,
                                     flags: Map[PredictResultFlag.Value, Boolean]) extends PredictResult(
  modelIdentifier, label, probability, significantFeatures, matchCount, flags
)

/**
  * Class to build a single label document prediction result.
  * @param modelIdentifier a string representing a model, right now the class name.
  * @param label
  */
class SingleLabelDocumentResultBuilder(modelIdentifier: String, label: String) {
  private var probability: Float = 0.0f
  private val significantFeatures = scala.collection.mutable.MutableList[(String, Float)]()
  private var matchCount: Int = 0
  private val specialFlags = mutable.HashMap[PredictResultFlag.Value, Boolean]()

  /**
    * Sets the probability.
    * @param probability
    * @return
    */
  def setProbability(probability: Float): SingleLabelDocumentResultBuilder = {
    this.probability = probability
    this
  }

  /**
    * Adds to the current match count.
    * @param matchCount
    * @return
    */
  def addToMatchCount(matchCount: Int): SingleLabelDocumentResultBuilder = {
    this.matchCount += matchCount
    this
  }

  /**
    * Sets the match count.
    * @param matchCount
    * @return
    */
  def setMatchCount(matchCount: Int): SingleLabelDocumentResultBuilder = {
    this.matchCount = matchCount
    this
  }

  /**
    * Appends a feature to this result.
    * @param feature
    * @return
    */
  def addSignificantFeature(feature: (String, Float)): SingleLabelDocumentResultBuilder = {
    this.significantFeatures += feature
    this
  }

  /**
    * Adds a whole list to this result.
    * @param significantFeatures
    * @return
    */
  def addSignificantFeatures(significantFeatures: List[(String, Float)]): SingleLabelDocumentResultBuilder = {
    this.significantFeatures ++= significantFeatures
    this
  }

  /**
    * Sets flags that we use for decision logic.
    * @param name
    * @param value
    * @return
    */
  def setFlags(name: PredictResultFlag.Value, value: Boolean): SingleLabelDocumentResultBuilder = {
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
