package com.idibon.ml.predict

/** Basic interfaces used for predictive analytics */
trait PredictModel[+T <: PredictResult] {
  /**
    * The method used to predict from a vector of features.
    * @param features Vector of features to use for prediction.
    * @param options Object of predict options.
    * @return
    */
  def predict(document: Document, options: PredictOptions): Seq[T]

  /**
    * The model will use a subset of features passed in. This method
    * should return the ones used.
    * @return Vector (likely SparseVector) where indices correspond to features
    *         that were used.
    */
  def getFeaturesUsed(): org.apache.spark.mllib.linalg.Vector
}
