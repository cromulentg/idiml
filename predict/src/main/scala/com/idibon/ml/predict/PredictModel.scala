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
    * The method used to predict from a vector of features.
    * @param features Vector of features to use for prediction.
    * @param significantFeatures whether to return significant features.
    * @param significantThreshold if returning significant features the threshold to use.
    * @return
    */
  def predict(features: Vector, significantFeatures: Boolean, significantThreshold: Double): PredictResult

  /**
    * The method used to predict from a FULL DOCUMENT!
    *
    * The model needs to handle "featurization" here.
    *
    * @param document the JObject to pull from.
    * @param significantFeatures whether to return significant features.
    * @param significantThreshold if returning significant features the threshold to use.
    * @return
    */
  def predict(document: JObject, significantFeatures: Boolean, significantThreshold: Double): PredictResult

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
