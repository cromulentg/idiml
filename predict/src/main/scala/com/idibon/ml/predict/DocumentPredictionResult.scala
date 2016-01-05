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

  val probabilities = mutable.HashMap[Int, Double]()
  val significantFeatures = mutable.HashMap[Int, List[(Int, Double)]]()

  /**
    * Add a prediction result for a document.
    * @param labelIndex
    * @param probability
    * @param labelSignificantFeatures
    */
  def addDocumentPredictResult(labelIndex: Int, probability: Double,
                               labelSignificantFeatures: List[(Int, Double)]): Unit = {
    probabilities.put(labelIndex, probability)
    if (labelSignificantFeatures != null && !labelSignificantFeatures.isEmpty) {
      significantFeatures.put(labelIndex, labelSignificantFeatures)
    }
  }

  /**
    * Returns the immutable DocumentPredictionResult.
    * @return the prediction result object.
    */
  def build(): DocumentPredictionResult = {
    new DocumentPredictionResult(probabilities.toMap, significantFeatures.toMap)
  }
}
