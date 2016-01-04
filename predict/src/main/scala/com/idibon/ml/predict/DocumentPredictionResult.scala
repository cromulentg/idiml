package com.idibon.ml.predict

import scala.collection.mutable

/**
  * This class houses a document prediction result.
  * E.g. housing results, as well as significant features.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>"
  *
  * @param probabilities immutable
  * @param significantFeatures immutable
  */
class DocumentPredictionResult(probabilities: Map[Int, Double],
                               significantFeatures: Map[Int, List[(Int, Double)]]) {

  override def toString: String = probabilities.toString()

  def size: Int = probabilities.size
}

class DocumentPredictionResultBuilder() {

  private val probabilities = mutable.HashMap[Int, Double]()
  private val significantFeatures = mutable.HashMap[Int, List[(Int, Double)]]()
  private var matchCount: Double = 0.0

  /**
    * Add a prediction result for a document.
    * @param labelIndex
    * @param probability
    * @param labelSignificantFeatures
    */
  def addDocumentPredictResult(labelIndex: Int, probability: Double,
                               labelSignificantFeatures: List[(Int, Double)]): DocumentPredictionResultBuilder = {
    probabilities.put(labelIndex, probability)
    if (labelSignificantFeatures != null && !labelSignificantFeatures.isEmpty) {
      significantFeatures.put(labelIndex, labelSignificantFeatures)
    }
    this
  }

  def setMatchCount(matchCount: Double): DocumentPredictionResultBuilder = {
    this.matchCount = matchCount
    this
  }

  /**
    * Returns the immutable DocumentPredictionResult.
    * @return the prediction result object.
    */
  def build(): DocumentPredictionResult = {
    new DocumentPredictionResult(probabilities.toMap, significantFeatures.toMap)
  }
}
