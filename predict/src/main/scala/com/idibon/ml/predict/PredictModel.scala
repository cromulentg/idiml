package com.idibon.ml.predict

import com.idibon.ml.feature.Archivable
import org.apache.spark.mllib.linalg.Vector
import org.json4s._


/**
  * "Interface" for prediction. Extends alloy reader and writer.
  *
  */
trait PredictModel extends Archivable {

  /**
    * The method used to predict.
    * @param features Vector of features to use for prediction.
    * @return Vector where the index corresponds to a label and the value the
    *         probability for that label.
    */
  def predict(features: Vector, significantFeatures: Boolean): DocumentPredictionResult

  /**
    * THe method used to predict FROM A DOCUMENT!
    * @param document
    * @param significantFeatures
    * @return
    */
  def predict(document: JObject, significantFeatures: Boolean): DocumentPredictionResult

  /**
    * Returns the type of model.
    * @return canonical class name.
    */
  def getType(): String

  /**
    * The model will use a subset of features passed in. This method
    * should return the ones used.
    * @return Vector (likely SparseVector) where indices correspond to features
    *         that were used.
    */
  def getFeaturesUsed(): Vector
}
