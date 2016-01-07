package com.idibon.ml.predict

import scala.collection.mutable

/**
  * This class houses a document prediction result.
  * E.g. housing results, as well as significant features.
  *
  * TBD: whether this is for a single label, or for multiple labels, or...
  * Also should this be a case class?
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>"
  *
  * @param probabilities immutable
  * @param significantFeatures immutable (they are strings, not feature indexes here)
  * @param matchCount immutable
  * @param flags immutable
  */
class DocumentPredictionResult(probabilities: Map[Int, Double],
                               significantFeatures: Map[Int, List[(String, Double)]],
                               matchCount: Double,
                               flags: Map[String, Boolean]) {

  override def toString: String = probabilities.toString()
  def size: Int = probabilities.size
  def getProbability(label: Int): Double = {
    probabilities.getOrElse(label, 0.0)
  }
  def getMatchCount(): Double = matchCount
  def getSignificantFeatures(label: Int): List[(String, Double)] = {
    significantFeatures.getOrElse(label, List())
  }
}

object DocumentPredictionResult {
  // flags for special cases.
  val WHITELIST_OR_BLACKLIST = "WhitelistOrBlacklist"
}

class DocumentPredictionResultBuilder() {
  private val probabilities = mutable.HashMap[Int, Double]()
  private val significantFeatures = mutable.HashMap[Int, List[(String, Double)]]()
  private var matchCount: Double = 0.0
  private val specialFlags = mutable.HashMap[String, Boolean]()

  /**
    * Add a prediction result for a document.
    * @param labelIndex
    * @param probability
    * @param labelSignificantFeatures
    */
  def addDocumentPredictResult(labelIndex: Int, probability: Double,
                               labelSignificantFeatures: List[(String, Double)]): DocumentPredictionResultBuilder = {
    probabilities.put(labelIndex, probability)
    if (labelSignificantFeatures != null && !labelSignificantFeatures.isEmpty) {
      significantFeatures.put(labelIndex, labelSignificantFeatures)
    }
    this
  }

  /**
    * Sets the match count. For Rule models this will be >= 1.0, else it will be 0.0.
    * @param matchCount
    * @return
    */
  def setMatchCount(matchCount: Double): DocumentPredictionResultBuilder = {
    this.matchCount = matchCount
    this
  }

  /**
    * Sets flags that we use for decision logic.
    * @param name
    * @param value
    * @return
    */
  def setFlags(name: String, value: Boolean): DocumentPredictionResultBuilder = {
    specialFlags.put(name, value)
    this
  }

  /**
    * Returns the immutable DocumentPredictionResult.
    * @return the prediction result object.
    */
  def build(): DocumentPredictionResult = {
    new DocumentPredictionResult(
      probabilities.toMap, significantFeatures.toMap, matchCount, specialFlags.toMap)
  }
}
