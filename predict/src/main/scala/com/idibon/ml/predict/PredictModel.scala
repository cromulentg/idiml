package com.idibon.ml.predict

import com.idibon.ml.predict.ml.TrainingSummary

/** Basic interfaces used for predictive analytics */
trait PredictModel[+T <: PredictResult] {

  /** Default training summary; None **/
  val trainingSummary: Option[Seq[TrainingSummary]] = None

  /**
    * The method used to predict from a vector of features.
    *
    * @param document Document that contains the original JSON.
    * @param options Object of predict options.
    * @return
    */
  def predict(document: Document, options: PredictOptions): Seq[T]

  /**
    * The model will use a subset of features passed in. This method
    * should return the ones used.
    *
    * @return Vector (likely SparseVector) where indices correspond to features
    *         that were used.
    */
  def getFeaturesUsed(): org.apache.spark.mllib.linalg.Vector

  /**
    * Returns a training summary. You have to override this to actually return something.
    * @return
    */
  def getTrainingSummary(): Option[Seq[TrainingSummary]] = trainingSummary
}
