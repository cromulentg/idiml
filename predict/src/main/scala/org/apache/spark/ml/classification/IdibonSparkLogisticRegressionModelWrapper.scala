package org.apache.spark.ml.classification;

import org.apache.spark.mllib.linalg.Vector

/**
  * This class extends LR so we can do predictions at the atomic
  * Vector level, since we want to bypass using DataFrames for
  * predicting on single feature vectors.
  *
  * @author Stefan Krawczyk <stefan@idibon.com>
  *
  *
  * @param uid
  * @param weights
  * @param intercept
  */
class IdibonSparkLogisticRegressionModelWrapper(override val uid: String,
                                                override val weights: Vector,
                                                override val intercept: Double)
  extends LogisticRegressionModel(uid, weights, intercept){

  /**
    * Makes this method Public for us to access.
    *
    * @param features features to use for prediction.
    * @return a Vector where each index corresponds to the label index,
    *         and the value there is the probability.
    *         e.g. a binary classifier will have two results returned.
    */
  override def predictProbability(features: Vector): Vector = {
    super.predictProbability(features)
  }
}
